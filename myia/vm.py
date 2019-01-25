"""Debug/Testing Virtual Machine.

This VM will directly execute a graph so it should be suitable for
testing or debugging.  Don't expect stellar performance from this
implementation.
"""

from typing import Iterable, Mapping, Any, List

from .ir import Graph, Apply, Constant, Parameter, ANFNode, MetaGraph
from .prim import Primitive
from .prim.py_implementations import typeof
from .prim.ops import return_, partial, embed
from .graph_utils import toposort
from .utils import TypeMap, is_dataclass_type, SymbolicKeyInstance


class VMFrame:
    """An execution frame.

    This holds the state for an application of a graph.  The todo list
    must contain free variables of graphs encountered before the
    graph themselves.

    You can index a frame with a node to get its value in the context
    of this frame (if it has already been evaluated).

    Attributes:
        values: Mapping of node to their values in this application
        todo: list of nodes remaining to execute
        closure: values for the closure if the current application is a closure

    """

    def __init__(self, nodes: Iterable[ANFNode], values: Mapping[ANFNode, Any],
                 *, closure: Mapping[ANFNode, Any] = None) -> None:
        """Initialize a frame."""
        self.values = dict(values)
        self.todo = list(nodes)
        self.todo.reverse()
        self.closure = closure

    def __getitem__(self, node: ANFNode):
        if node in self.values:
            return self.values[node]
        elif self.closure is not None and node in self.closure:
            return self.closure[node]
        elif node.is_constant():
            # Should be a constant
            return node.value
        else:
            raise ValueError(node)  # pragma: no cover


class Closure:
    """Representation of a closure."""

    def __init__(self, graph: Graph, values: Mapping[ANFNode, Any]) -> None:
        """Build a closure."""
        self.graph = graph
        self.values = values
        self.vm: 'VM' = None

    def __call__(self, *args):
        """Evaluates the closure."""
        return self.vm.evaluate(self.graph, args, closure=self.values)


class Partial:
    """Representation of a partial application."""

    def __init__(self, fn, args, vm):
        """Build a partial."""
        self.fn = fn
        self.args = tuple(args)
        self.vm = vm

    def __call__(self, *args):
        """Evaluates the partial."""
        return self.vm.call(self.fn, self.args + args)


class VM:
    """Virtual Machine interface."""

    class _Call(Exception):
        """Indicate a call to a new frame."""

        def __init__(self, frame):
            self.frame = frame

    class _Return(Exception):
        """Indicates a return with its value."""

        def __init__(self, value):
            self.value = value

    def __init__(self, convert, manager, py_implementations, implementations):
        """Initialize the VM."""
        self.convert = convert
        self.manager = manager
        self._exporters = TypeMap({
            tuple: self._export_sequence,
            list: self._export_sequence,
            Closure: self._export_Closure,
            Graph: self._export_Graph,
            Primitive: self._export_Primitive,
            object: self._export_object,
        })
        self.implementations = implementations
        self.py_implementations = py_implementations
        self._vars = dict()

    def _compute_fvs(self, graph):
        rval = set()
        for fv in graph.free_variables_total:
            if isinstance(fv, Graph):
                rval.update(graph.manager.graph_constants[fv])
            else:
                rval.add(fv)
        return rval

    def _acquire_graph(self, graph):
        if graph in self._vars:
            return
        self.manager.add_graph(graph)
        for g in graph.manager.graphs:
            self._vars[g] = self._compute_fvs(g)

    def _export_sequence(self, seq):
        return type(seq)(self.export(x) for x in seq)

    def _export_Primitive(self, prim):
        return self.py_implementations[prim]

    def _export_Closure(self, clos):
        clos.vm = self
        return clos

    def _export_Graph(self, g):
        """Return an object that executes `g` when called on arguments."""
        c = Closure(g, None)
        c.vm = self
        return c

    def _export_object(self, obj):
        return obj

    def export(self, value):
        """Convert a value from the VM into a corresponding Python object."""
        return self._exporters[type(value)](value)

    def evaluate(self, graph: Graph, _args: Iterable[Any], *,
                 closure: Mapping[ANFNode, Any] = None) -> Any:
        """Run a graph.

        This will evaluate the passed-in graph and return the
        resulting value.
        """
        args = self.convert(tuple(_args))

        self._acquire_graph(graph)

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        top_frame = VMFrame(toposort(graph.return_, self._succ_vm(graph)),
                            dict(zip(graph.parameters, args)),
                            closure=closure)
        frames = [top_frame]

        while frames:
            try:
                frame = frames[-1]
                todo = frame.todo
                while todo:
                    self._handle_node(todo[-1], frame)
                    todo.pop()
            except self._Call as c:
                # The last element of todo is always a return
                if len(todo) == 2:
                    frames[-1] = c.frame
                else:
                    frames.append(c.frame)
            except self._Return as r:
                frames.pop()
                if frames:
                    frames[-1].values[frames[-1].todo[-1]] = r.value
                    frames[-1].todo.pop()
                else:
                    return self.export(r.value)

    def _succ_vm(self, graph):
        """Return a visitor for the graph."""
        def succ(node: ANFNode) -> Iterable[ANFNode]:
            """Follow node.incoming and free variables."""
            for i in node.inputs:
                if (i.graph == node.graph or
                        i.is_constant_graph() and i.value.parent == graph):
                    yield i
            if node.is_constant_graph() and node.value.parent == graph:
                yield from self._vars[node.value]
        return succ

    def call(self, fn, args):
        """Call the `fn` object.

        `fn` can be anything that would be valid as the first element
        of an apply.
        """
        if isinstance(fn, Primitive):
            return self.implementations[fn](self, *args)

        elif isinstance(fn, Graph):
            return self.evaluate(fn, args)

        elif isinstance(fn, Closure):
            return self.evaluate(fn.graph, args, closure=fn.values)

        else:
            raise AssertionError(f"Can't call {fn}")

    def _call(self, graph: Graph, args: List[Any]):
        clos = None
        if isinstance(graph, Closure):
            clos = graph.values
            graph = graph.graph

        assert isinstance(graph, Graph)

        if graph not in self._vars:
            self._acquire_graph(graph)

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        raise self._Call(VMFrame(toposort(graph.return_, self._succ_vm(graph)),
                                 dict(zip(graph.parameters, args)),
                                 closure=clos))

    def _make_closure(self, graph: Graph, frame: VMFrame) -> Closure:
        clos = dict()
        for v in self._vars[graph]:
            clos[v] = frame[v]
        return Closure(graph, clos)

    def _dispatch_call(self, node, frame, fn, args):
        if isinstance(fn, Primitive):
            if fn == return_:
                raise self._Return(args[0])
            elif fn == partial:
                partial_fn, *partial_args = args
                res = Partial(partial_fn, partial_args, self)
                frame.values[node] = res
            elif fn == embed:
                _, x = node.inputs
                frame.values[node] = SymbolicKeyInstance(x, {})
            else:
                frame.values[node] = self.implementations[fn](self, *args)
        elif isinstance(fn, Partial):
            self._dispatch_call(node, frame, fn.fn, fn.args + tuple(args))
        elif isinstance(fn, (Graph, Closure)):
            self._call(fn, args)
        elif isinstance(fn, MetaGraph):
            types = [typeof(arg) for arg in args]
            g = fn.specialize_from_types(types)
            self._dispatch_call(node, frame, g, args)
        elif is_dataclass_type(fn):
            frame.values[node] = fn(*args)
        else:
            raise AssertionError(f'Invalid fn to call: {fn}')

    def _handle_node(self, node: ANFNode, frame: VMFrame):
        if isinstance(node, Constant):
            # We only visit constant graphs
            assert node.is_constant_graph()
            if frame.closure is not None and node in frame.closure:
                return
            g = node.value
            if len(self._vars[g]) != 0:
                frame.values[node] = self._make_closure(g, frame)
            # We don't need to do anything special for non-closures

        elif isinstance(node, Parameter):
            pass

        elif isinstance(node, Apply):
            fn, *args = (frame[inp] for inp in node.inputs)
            self._dispatch_call(node, frame, fn, args)

        else:
            raise AssertionError("Unknown node type")  # pragma: no cover
