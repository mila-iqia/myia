"""Transforms a graph into lower-level code."""

from ..abstract import (
    AbstractArray,
    AbstractHandle,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractType,
    to_abstract,
)
from ..ir import Apply, Constant, Graph, toposort
from ..operations import Primitive, primitives as P
from ..utils import HandleInstance, SymbolicKeyInstance, overload
from .channel import handle
from .vm import FinalVM


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
        if node.is_constant(Primitive):
            if node.value is P.env_setitem:
                node.abstract = None
            if node.value is P.env_getitem:
                node.abstract = None

    return graph


def get_prim_graph(cache, prim, typ):
    """Make a graph that wraps a primitive."""
    if (prim, typ) not in cache:
        g = Graph()
        args = []
        tp = list(typ.xvalue())[0]
        for t in tp.args:
            p = g.add_parameter()
            p.abstract = t
            args.append(p)
        primct = Constant(prim)
        primct.abstract = typ
        out = g.apply(primct, *args)
        out.abstract = tp.output
        g.output = out
        cache[(prim, typ)] = g
    return cache[(prim, typ)]


def wrap_primitives(graph):
    """Helper function to wrap primitives.

    This wraps all primitives used in non-call positions in a graph.
    """
    mng = graph.manager

    prim_graphs = {}

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
                        g = get_prim_graph(prim_graphs, ct.value, ct.abstract)
                        tr.set_edge(node, key, Constant(g))

    return graph


def gather_handles_cst(node):
    """Return a list of constant handles."""
    lst = []
    if node.is_constant(HandleInstance):
        lst.append((node.value, node))
    return lst


@overload(bootstrap=True)
def _gather_handles_param(self, t: AbstractHandle, curnode, idx, curget, lst):
    lst.append((idx, curget, curnode))


@overload  # noqa: F811
def _gather_handles_param(self, t: AbstractTuple, curnode, idx, curget, lst):
    for i, tt in enumerate(t.elements):
        self(tt, (P.tuple_getitem, curnode, i), idx, lambda v, i=i: v[i], lst)


@overload  # noqa: F811
def _gather_handles_param(self,
                          t: (AbstractArray, AbstractScalar,
                              AbstractType, AbstractTaggedUnion),
                          *args):
    return


def gather_handles_params(params):
    """
    Returns a list of all the handles in the parameters with graph
    and value accessors.
    """
    lst = []
    for i, p in enumerate(params):
        _gather_handles_param(p.abstract, p, i, lambda v: v, lst)
    return lst


def return_handles(graph):
    """Change the Universe output to return all the new values of handles."""
    mng = graph.manager

    handles_params = gather_handles_params(graph.parameters)

    cts = {ct for cts in mng.constants.values() for ct in cts}
    for ct in cts:
        handles_cst = gather_handles_cst(ct)

    handle_nodes = [h for _, h in handles_cst]
    handle_nodes.extend(sexp_to_node(n, graph) for _, _, n in handles_params)

    with mng.transact() as tr:
        universe_out = graph.output.inputs[1]
        vals = [graph.apply(P.universe_getitem, universe_out, n)
                for n in handle_nodes]
        tr.set_edge(graph.output, 1, graph.apply(P.make_tuple, *vals))

    return (graph, [i for i, _ in handles_cst],
            [(i, get) for i, get, _ in handles_params])


def handle_wrapper(fn, handle_cst, handle_params):
    """Wraps a model function to perform handle updates."""
    def wrapper(*args):
        handle_instances = handle_cst
        handle_instances.extend(get(args[i]) for i, get in handle_params)
        res = fn(*args)
        u = res[0]
        res = res[1] if len(res) == 2 else res[1:]
        for h, v in zip(handle_instances, u):
            h.state = v
        return res
    if len(handle_cst) + len(handle_params) == 0:
        return fn
    else:
        return wrapper


@overload
def wrap_result(data: tuple):
    """Function to wrap final results in a handle.

    This leaves first-level tuples alone so that we support multiple
    value returns more naturally.
    """
    return tuple(handle(d) for d in data)


@overload  # noqa: F811
def wrap_result(data: object):
    return handle(data)


nonlinear_ops = (
    P.return_, P.partial, P.switch, P.make_tuple, P.bool_and,
    P.tuple_getitem, P.tuple_setitem, P.env_getitem, P.env_setitem, P.env_add,
    P.tagged, P.hastag, P.casttag, P.unsafe_static_cast,
)


class CompileGraph:
    """Helper to compile a graph to a linear set of instructions.

    Inputs:
        graph: A graph
        splits: list of graph portions

    Outputs:
        uinstrs: list of instructions for the graph (unlinked)

    """

    def __init__(self, lin_convert, cut_list, backend):
        """Create a CompileGraph with the specified linear backend."""
        self.lin_convert = lin_convert
        self.cut_list = cut_list
        self.backend = backend

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

        for node in toposort(graph.return_):
            if self._is_cut(node):
                splits.append(node)
            elif not (node.is_constant() or node.is_parameter()):
                splits.append([node])

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
                v = self.backend.to_backend_value(node.value, node.abstract)
                self.add_instr('push', v)
            self.push(node)
        return self.slots[node] - self.height

    def dup(self, node):
        """Ensures that the value for node is at the top of the stack."""
        assert node in self.slots
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
                    elif fn.value == P.switch:
                        self.add_instr('switch', self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.make_tuple:
                        self.add_instr('tuple', *[self.ref(i)
                                                  for i in split.inputs[1:]])
                    elif fn.value == P.bool_and:
                        self.add_instr('bool_and',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.tuple_getitem:
                        self.add_instr('tuple_getitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.tuple_setitem:
                        self.add_instr('tuple_setitem',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]),
                                       self.ref(split.inputs[3]))
                    elif fn.value == P.tagged:
                        self.add_instr('tagged',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.hastag:
                        self.add_instr('hastag',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.casttag:
                        self.add_instr('casttag',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
                    elif fn.value == P.unsafe_static_cast:
                        self.add_instr('unsafe_static_cast',
                                       self.ref(split.inputs[1]),
                                       self.ref(split.inputs[2]))
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

    def __init__(self, lin_convert, cut_list, backend):
        """Create a compiler.

        This use the specifed implementation for linear parts and a
        list of excluded ops that will be covered by the built-in VM.

        """
        self.transform = CompileGraph(lin_convert, cut_list, backend)
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


__all__ = [
    'CompileGraph',
    'CompileGraphs',
    'convert_grad',
    'wrap_primitives',
    'wrap_result',
    'handle_wrapper',
    'return_handles',
]
