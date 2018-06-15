from ..ir import (Apply, is_apply, is_constant, is_constant_graph,
                  is_parameter, toposort)
from ..pipeline import PipelineDefinition, PipelineStep
from ..prim import Primitive
from ..prim.ops import if_, partial, return_
from .debug_lin import debug_convert
from .vm import FinalVM


class SplitGraph(PipelineStep):
    """Pipeline stop to cut the graph into linear portions and control flow."""

    def step(self, graph):
        splits = []
        split = []

        for node in toposort(graph.return_):
            if self._cut(node):
                if len(split) != 0:
                    splits.append(split)
                splits.append(node)
                split = []
            elif not (is_constant(node) or is_parameter(node)):
                split.append(node)

        return {'splits': splits}

    def _cut(self, node):
        if is_apply(node):
            fn = node.inputs[0]
            if not is_constant(fn, Primitive):
                return True
            elif fn.value in (if_, return_, partial):
                return True
        return False


class CompileGraph(PipelineStep):
    """Step to convert splits into linear instruction flow."""

    def __init__(self, pipeline_init, lin_convert):
        super().__init__(pipeline_init)
        self.lin_convert = lin_convert

    def _reset(self):
        self._height = 0
        self.max_height = 0
        self.slots = {}
        self.instrs = []

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, val):
        self._height = val
        self.max_height = max(self.max_height, self._height)

    def add_instr(self, instr, *args):
        self.instrs.append((instr,) + args)

    def push(self, node):
        assert node not in self.slots
        self.slots[node] = self.height
        self.height += 1

    def ref(self, node):
        if node not in self.slots and is_constant(node):
            if is_constant_graph(node):
                self.add_instr('push_graph', node.value)
            else:
                self.add_instr('push', node.value)
            self.push(node)
        return self.slots[node] - self.height

    def dup(self, node):
        if node not in self.slots:
            return self.ref(node)
        self.add_instr('dup', self.ref(node))
        self.height += 1
        return -1

    def ret(self, nargs):
        self.height -= nargs

    def step(self, graph, splits):
        self._reset()

        for p in reversed(graph.parameters):
            self.push(p)

        param_height = self.height

        for split in splits:
            if isinstance(split, list):
                run, inputs, outputs = self.lin_convert(split)
                args = [self.ref(i) for i in inputs]
                self.add_instr('lin_apply', run, args)
                for o in outputs:
                    self.push(o)

            else:
                assert isinstance(split, Apply)
                fn = split.inputs[0]

                if is_constant(fn, Primitive):
                    # pre-push arguments on the stack if needed
                    for i in split.inputs[1:]:
                        self.ref(i)
                    if fn.value == if_:
                        if split is graph.output:
                            self.add_instr('tailif', self.ref(split.inputs[1]),
                                           self.ref(split.inputs[2]),
                                           self.ref(split.inputs[3]),
                                           self.height)
                            # execution stops here
                            break
                        else:
                            self.add_instr('if', self.ref(split.inputs[1]),
                                           self.ref(split.inputs[2]),
                                           self.ref(split.inputs[3]))
                    elif fn.value == return_:
                        self.add_instr('return', self.ref(split.inputs[1]),
                                       self.height)
                        # execution stops here
                        break
                    elif fn.value == partial:
                        self.add_instr(
                            'partial', self.ref(split.inputs[1]),
                            *tuple(self.ref(inp) for inp in split.inputs[2:]))
                    else:
                        raise AssertionError(f"Unknown special function "
                                             "{fn.value}")

                else:
                    # pre-push the function on the stack
                    self.ref(fn)
                    for i in reversed(split.inputs[1:]):
                        self.dup(i)
                    if split is graph.output:
                        self.add_instr('tailcall', self.ref(fn), self.height,
                                       len(split.inputs[1:]))
                        # execution stops here
                        break
                    else:
                        self.add_instr('call', self.ref(fn))
                        self.ret(len(split.inputs))

                self.push(split)

        need_stack = self.max_height - param_height
        if need_stack > 0:
            self.instrs.insert(0, ('pad_stack', need_stack))

        res = {'instrs': self.instrs}
        self._reset()
        return res


class OptimizeInstrs(PipelineStep):
    def step(self, instrs):
        return {'instrs': instrs}


graph_transform = PipelineDefinition(
    steps=dict(
        split=SplitGraph.partial(),
        compile=CompileGraph.partial(lin_convert=debug_convert),
        optimize=OptimizeInstrs.partial(),
    )
)


class CompileGraphs(PipelineStep):
    def __init__(self, pipeline_init, transform):
        super().__init__(pipeline_init)
        self.transform = transform

    def reset(self):
        self.mapping = {}
        self.instrs = []

    def compile(self, graph):
        self.mapping[graph] = len(self.instrs)
        self.instrs.extend(self.transform(graph=graph)['instrs'])

    def step(self, graph):
        self.reset()

        self.compile(graph)

        graphs = graph.manager.graphs
        for g in (graphs - set([graph])):
            self.compile(g)

        res = {'mapping': self.mapping, 'instrs': self.instrs}
        self.reset()
        return res


class LinkInstrs(PipelineStep):
    def step(self, mapping, instrs):
        for i in range(len(instrs)):
            instr = instrs[i]
            if instr[0] == 'push_graph':
                instrs[i] = ('push', mapping[instr[1]])

        return {'instrs': instrs}


class VMExporter(PipelineStep):
    def step(self, instrs):
        return {'output': FinalVM(instrs)}


step_compile = CompileGraphs.partial(transform=graph_transform.make())
step_link = LinkInstrs.partial()
step_export = VMExporter.partial()
