"""Transforms a graph into lower-level code."""

from ..ir import Apply, toposort, Graph, Constant
from ..pipeline import PipelineDefinition, PipelineStep
from ..prim import Primitive, ops as P
from ..prim.ops import partial, return_, switch, make_tuple
from .debug_lin import debug_convert
from .nnvm import nnvm_convert
from .vm import FinalVM

LIN_IMPLS = dict(
    debug=debug_convert,
    nnvm=nnvm_convert,
)


class WrapPrimitives(PipelineStep):
    """Pipeline step to wrap primitives in non-call positions into graphs.

    Inputs:
        graph: A graph

    Outputs:
        graph: The transformed graph

    """

    def step(self, graph):
        """Wrap primitives in non-call positions into graphs."""
        mng = self.resources.manager
        mng.add_graph(graph)

        prim_graphs = {}

        def get_prim_graph(prim, typ):
            if (prim, typ) not in prim_graphs:
                g = Graph()
                args = []
                for t in typ.arguments:
                    p = g.add_parameter()
                    p.type = t
                    args.append(p)
                primct = Constant(prim)
                primct.type = typ
                out = g.apply(primct, *args)
                out.type = typ.retval
                g.output = out
                prim_graphs[(prim, typ)] = g
            return prim_graphs[(prim, typ)]

        with mng.transact() as tr:
            cts = {ct for cts in mng.constants.values() for ct in cts}
            for ct in cts:
                if ct.is_constant(Primitive):
                    for node, key in mng.uses[ct]:
                        if key != 0:
                            if node.inputs[0].is_constant():
                                if node.inputs[0].value in (P.array_map,
                                                            P.array_reduce):
                                    continue
                            g = get_prim_graph(ct.value, ct.type)
                            tr.set_edge(node, key, Constant(g))

        return {'graph': graph}


class SplitGraph(PipelineStep):
    """Pipeline step to cut the graph into linear portions and control flow.

    Inputs:
        graph: A graph

    Outputs:
        splits: list of graph portions

    """

    def step(self, graph):
        """Split the graph into portions."""
        splits = []
        split = []

        for node in toposort(graph.return_):
            if self.is_cut(node):
                if len(split) != 0:
                    splits.append(split)
                splits.append(node)
                split = []
            elif not (node.is_constant() or node.is_parameter()):
                split.append(node)

        return {'splits': splits}

    def is_cut(self, node):
        """Returns whether there should be a cut for this node.

        Cuts are done for all "non-linear" nodes: function calls,
        branches, ...

        """
        if node.is_apply():
            fn = node.inputs[0]
            if not fn.is_constant(Primitive):
                return True
            elif fn.value in (return_, partial, switch, make_tuple):
                return True
        return False


class CompileGraph(PipelineStep):
    """Step to convert splits into linear instruction flow.

    Inputs:
        graph: A graph
        splits: list of graph portions

    Outputs:
        uinstrs: list of instructions for the graph (unlinked)

    """

    def __init__(self, pipeline_init):
        """Initialize a the CompileGraph step."""
        super().__init__(pipeline_init)

    def _reset(self):
        """Set/clear shared values."""
        self._height = 0
        self.max_height = 0
        self.slots = {}
        self.instrs = []

    @property
    def height(self):
        """The current stack height."""
        return self._height

    @height.setter
    def height(self, val):
        self._height = val
        self.max_height = max(self.max_height, self._height)

    def add_instr(self, instr, *args):
        """Append instruction to the list."""
        self.instrs.append((instr,) + args)

    def push(self, node):
        """Simulate pushing the value for node on the stack.

        This records the position so that other nodes can refer to
        this value later.

        """
        assert node not in self.slots
        self.slots[node] = self.height
        self.height += 1

    def tie(self, n1, n2):
        """Declare two nodes as equivalent."""
        self.slots[n2] = self.slots[n1]

    def ref(self, node):
        """Get the stack reference for the value of a node.

        This can actually cause a push if the node is a constant that
        wasn't referred to before.

        """
        if node not in self.slots and node.is_constant():
            if node.is_constant_graph():
                self.add_instr('push_graph', node.value)
            else:
                assert not isinstance(node.value, Primitive)
                self.add_instr('push', node.value)
            self.push(node)
        return self.slots[node] - self.height

    def dup(self, node):
        """Ensures that the value for node is at the top of the stack."""
        if node not in self.slots:
            return self.ref(node)
        self.add_instr('dup', self.ref(node))
        self.height += 1
        return -1

    def ret(self, nargs):
        """Simulate the effect of a return from a call on the stack."""
        self.height -= nargs

    def step(self, graph, splits):
        """Convert the graph into a list of instructions."""
        self._reset()

        for p in reversed(graph.parameters):
            self.push(p)

        param_height = self.height

        for split in splits:
            if isinstance(split, list):
                run, inputs, outputs = \
                    self.pipeline.resources.lin_convert(
                        split,
                        target=self.pipeline.resources.target,
                        dev_id=self.pipeline.resources.dev_id)
                if run is None:  # empty function
                    assert len(inputs) == len(outputs)
                    for i, o in zip(inputs, outputs):
                        self.tie(i, o)
                    continue
                # prime the arguments because self.ref() can invalidate
                # previously returned references if a new one is not ready
                for i in inputs:
                    self.ref(i)
                args = [self.ref(i) for i in inputs]
                self.add_instr('external', run, args)
                for o in outputs:
                    self.push(o)

            else:
                assert isinstance(split, Apply)
                fn = split.inputs[0]

                if fn.is_constant(Primitive):
                    # prime the arguemnts because self.ref() can invalidate
                    # previously returned references if a new one is not ready
                    for i in split.inputs[1:]:
                        self.ref(i)
                    if fn.value == return_:
                        self.add_instr('return', self.ref(split.inputs[1]),
                                       self.height)
                        # execution stops here
                        break
                    elif fn.value == partial:
                        self.add_instr(
                            'partial', self.ref(split.inputs[1]),
                            *tuple(self.ref(inp) for inp in split.inputs[2:]))
                    elif fn.value == switch:  # pragma: no cover
                        self.add_instr('switch', self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))

                    elif fn.value == make_tuple:
                        self.add_instr('tuple', *[self.ref(i)
                                                  for i in split.inputs[1:]])
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
                        self.ret(len(split.inputs) - 1)

                self.push(split)

        need_stack = self.max_height - param_height
        if need_stack > 0:
            self.instrs.insert(0, ('pad_stack', need_stack))

        res = {'uinstrs': self.instrs}
        self._reset()
        return res


class OptimizeInstrs(PipelineStep):
    """Run peephole optimizations.

    Inputs:
        uinstrs: List of unlinked instructions

    Outputs:
        uinstrs: List of unlinked instructions
    """

    def step(self, uinstrs):
        """Apply optimizations (currently none)."""
        return {'uinstrs': uinstrs}


graph_transform = PipelineDefinition(
    resources=dict(
        lin_convert=nnvm_convert,
        target='cpu',
        dev_id=0,
    ),
    steps=dict(
        split=SplitGraph.partial(),
        compile=CompileGraph.partial(),
        optimize=OptimizeInstrs.partial(),
    )
)


class CompileGraphs(PipelineStep):
    """Convert a graph cluster into instruction lists.

    Inputs:
        graph: A graph

    Outputs:
        mapping: map each graph to its starting position in the code list.
        uinstrs: list of unlinked instructions for all the graphs in
                 the cluster, starting with the passed-in graph.

    """

    def __init__(self, pipeline_init, linear_impl, target, dev_id):
        """Initialize a CompileGraphs.

        Arguments:
            linear_impl: the implementation to use for linear parts.

        """
        super().__init__(pipeline_init)
        self.transform = graph_transform.configure(
            lin_convert=LIN_IMPLS[linear_impl],
            target=target,
            dev_id=dev_id).make()

    def reset(self):
        """Clear/set local variables."""
        self.mapping = {}
        self.instrs = []

    def compile(self, graph):
        """Convert a single graph to unlinked instructions and map it."""
        self.mapping[graph] = len(self.instrs)
        self.instrs.extend(self.transform(graph=graph)['uinstrs'])

    def step(self, graph):
        """Convert all graphs to unlinked instructions and map them."""
        self.reset()

        self.compile(graph)

        graphs = graph.manager.graphs
        for g in (graphs - set([graph])):
            self.compile(g)

        res = {'mapping': self.mapping, 'uinstrs': self.instrs}
        self.reset()
        return res


class LinkInstrs(PipelineStep):
    """Link unlinked instructions.

    Inputs:
        mapping: graph map
        uinstrs: unlinked instructions

    Outputs:
        instrs: linked instructions

    """

    def step(self, mapping, uinstrs):
        """Link instructions."""
        for i in range(len(uinstrs)):
            instr = uinstrs[i]
            if instr[0] == 'push_graph':
                uinstrs[i] = ('push', mapping[instr[1]])

        return {'instrs': uinstrs}


class VMExporter(PipelineStep):
    """Make a callable out of instructions.

    Inputs:
        instrs: instruction list

    Outputs:
        output: callable
    """

    def step(self, instrs):
        """Make a callable."""
        return {'output': FinalVM(instrs)}


step_wrap_primitives = WrapPrimitives.partial()
step_compile = CompileGraphs.partial(
    linear_impl='debug', target='cpu', dev_id=0)
step_link = LinkInstrs.partial()
step_export = VMExporter.partial()
