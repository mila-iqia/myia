"""Transforms a graph into lower-level code."""

from ..abstract import VALUE, to_abstract
from ..ir import Apply, toposort, Graph, Constant
from ..prim import Primitive, ops as P
from .vm import FinalVM
from ..utils import SymbolicKeyInstance


def convert_grad(graph):
    """Remove all instances of SymbolicKeyType in the graphs.

    They will be replaced by globally-unique integers.
    """
    mng = graph.manager

    counter = 0
    key_map = {}

    for node in mng.all_nodes:
        if node.is_constant(SymbolicKeyInstance):
            if node.value not in key_map:
                key_map[node.value] = counter
                counter += 1
            node.value = key_map[node.value]
            node.abstract = to_abstract(node.value)

    return graph


def wrap_primitives(graph):
    """Helper function to wrap primitives.

    This wraps all primitives used in non-call positions in a graph.
    """
    mng = graph.manager

    prim_graphs = {}

    def get_prim_graph(prim, typ):
        if (prim, typ) not in prim_graphs:
            g = Graph()
            args = []
            tp = list(typ.values[VALUE])[0]
            for t in tp.args:
                p = g.add_parameter()
                p.abstract = t
                args.append(p)
            primct = Constant(prim)
            primct.abstract = typ
            out = g.apply(primct, *args)
            out.abstract = tp.output
            g.output = out
            prim_graphs[(prim, typ)] = g
        return prim_graphs[(prim, typ)]

    with mng.transact() as tr:
        cts = {ct for cts in mng.constants.values() for ct in cts}
        for ct in cts:
            if ct.is_constant(Primitive):
                for node, key in mng.uses[ct]:
                    if key != 0:
                        if (key == 1 and
                            node.inputs[0].is_constant() and
                            node.inputs[0].value in (P.array_map,
                                                     P.array_reduce)):
                            continue
                        g = get_prim_graph(ct.value, ct.abstract)
                        tr.set_edge(node, key, Constant(g))

    return graph


nonlinear_ops = (
    P.return_, P.partial, P.switch, P.make_tuple, P.make_list,
    P.list_len, P.list_getitem, P.list_setitem, P.list_append, P.bool_and,
    P.tuple_getitem, P.tuple_setitem, P.env_getitem, P.env_setitem, P.env_add
)


class CompileGraph:
    """Helper to compile a graph to a linear set of instructions.

    Inputs:
        graph: A graph
        splits: list of graph portions

    Outputs:
        uinstrs: list of instructions for the graph (unlinked)

    """

    def __init__(self, lin_convert, cut_list, backend, *, split_linear=False):
        """Create a CompileGraph with the specified linear backend."""
        self.lin_convert = lin_convert
        self.cut_list = cut_list
        self.backend = backend
        self.split_linear = split_linear

    def _reset(self):
        """Set/clear shared values."""
        self._height = 0
        self.max_height = 0
        self.slots = {}
        self.instrs = []
        self.env_keys = []

    def _is_cut(self, node):
        if node.is_apply():
            fn = node.inputs[0]
            if not fn.is_constant(Primitive):
                return True
            elif fn.value in self.cut_list:
                return True
        return False

    def split(self, graph):
        """Split a graph into portions."""
        splits = []
        split = []

        for node in toposort(graph.return_):
            if self._is_cut(node):
                if len(split) != 0:
                    splits.append(split)
                splits.append(node)
                split = []
            elif not (node.is_constant() or node.is_parameter()):
                if self.split_linear:
                    splits.append([node])
                else:
                    split.append(node)

        return splits

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
                v = self.backend.convert_value(node.value, node.abstract)
                self.add_instr('push', v)
            self.push(node)
        return self.slots[node] - self.height

    def dup(self, node):
        """Ensures that the value for node is at the top of the stack."""
        if node not in self.slots:
            # This used to be covered by test_compile::test_tailcall
            return self.ref(node)  # pragma: no cover
        self.add_instr('dup', self.ref(node))
        self.height += 1
        return -1

    def ret(self, nargs):
        """Simulate the effect of a return from a call on the stack."""
        self.height -= nargs

    def run(self, graph):
        """Convert the graph into a list of instructions."""
        self._reset()

        splits = self.split(graph)

        for p in reversed(graph.parameters):
            self.push(p)

        param_height = self.height

        for split in splits:
            if isinstance(split, list):
                run, inputs, outputs = self.lin_convert(split)
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
                    if fn.value == P.return_:
                        self.add_instr('return', self.ref(split.inputs[1]),
                                       self.height)
                        # execution stops here
                        break
                    elif fn.value == P.partial:
                        self.add_instr(
                            'partial', self.ref(split.inputs[1]),
                            *tuple(self.ref(inp) for inp in split.inputs[2:]))
                    elif fn.value == P.switch:  # pragma: no cover
                        self.add_instr('switch', self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.make_tuple:
                        self.add_instr('tuple', *[self.ref(i)
                                                  for i in split.inputs[1:]])
                    elif fn.value == P.make_list:
                        self.add_instr('list', *[self.ref(i)
                                                 for i in split.inputs[1:]])
                    elif fn.value == P.list_len:
                        self.add_instr('list_len', self.ref(split.inputs[1]))
                    elif fn.value == P.list_getitem:
                        self.add_instr('list_getitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.list_setitem:
                        self.add_instr('list_setitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.list_append:
                        self.add_instr('list_append',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.bool_and:
                        self.add_instr('bool_and',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.tuple_getitem:
                        self.add_instr('tuple_getitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.tuple_setitem:  # pragma: no cover
                        raise RuntimeError(
                            "Please report your testcase so that "
                            "we can add it to the testsuite")
                        self.add_instr('tuple_setitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.env_getitem:
                        self.add_instr('env_getitem',
                                       self.ref(split.inputs[1]),
                                       split.inputs[2].value,
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.env_setitem:
                        self.add_instr('env_setitem',
                                       self.ref(split.inputs[1]),
                                       split.inputs[2].value,
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.env_add:  # pragma: no cover
                        raise RuntimeError("apparently no model requires this")
                        self.add_instr('env_add',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    else:
                        raise AssertionError(f"Unknown special function "
                                             "{fn.value}")

                else:
                    # ensure the function and arguments are available.
                    self.ref(fn)
                    for i in split.inputs[1:]:
                        self.ref(i)
                    # make references to the arguments
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

        res = self.instrs
        self._reset()
        return res


class CompileGraphs:
    """Convert a graph cluster into instruction lists.

    Inputs:
        graph: A graph

    Outputs:
        output: A callable

    """

    def __init__(self, lin_convert, cut_list, backend, *, split_linear=False):
        """Create a compiler.

        This use the specifed implementation for linear parts and a
        list of excluded ops that will be covered by the built-in VM.

        """
        self.transform = CompileGraph(lin_convert, cut_list, backend,
                                      split_linear=split_linear)
        self._reset()

    def _reset(self):
        self.mapping = {}
        self.instrs = []

    def compile(self, graph):
        """Convert a single graph to unlinked instructions and map it."""
        self.mapping[graph] = len(self.instrs)
        self.instrs.extend(self.transform.run(graph=graph))

    def link(self):
        """Link instructions from multiple graphs together."""
        for i in range(len(self.instrs)):
            instr = self.instrs[i]
            if instr[0] == 'push_graph':
                self.instrs[i] = ('push', self.mapping[instr[1]])

    def compile_and_link(self, graph):
        """Convert all graphs to unlinked instructions and map them."""
        self._reset()

        graph = wrap_primitives(graph)
        graph = convert_grad(graph)

        self.compile(graph)

        graphs = graph.manager.graphs
        for g in (graphs - set([graph])):
            self.compile(g)

        self.link()

        res = FinalVM(self.instrs, self.transform.backend)
        self._reset()
        return res
