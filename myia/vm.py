"""Debug/Testing Virtual Machine.

This VM will directly execute a graph so it should be suitable for
testing or debugging.  Don't expect stellar performance from this
implementation.
"""
from types import FunctionType
from typing import Iterable, Mapping, Any, Callable, List

from myia.ir.anf import Graph, Apply, Constant, Parameter, ANFNode
from myia.ir.utils import is_constant_graph
from myia.prim import Primitive
from myia.prim.ops import if_, return_
from myia.utils import smap
from myia.graph_utils import toposort
from myia.cconv import NestingAnalyzer


class VMFrame:
    """An execution frame.

    This holds the state for an application of a graph.  The todo list
    must contain free variables of graphs encountered before the
    graph themselves.

    You can index a frame with a node to get its value in the context
    of this frame (if it has already been evaluated).

    Attributes
    ----------
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
        else:
            # Should be a constant
            return node.value


class Closure:
    """Representation of a closure."""

    def __init__(self, graph: Graph, values: Mapping[ANFNode, Any]) -> None:
        """Build a closure."""
        self.graph = graph
        self.values = values
        self.vm: 'VM' = None

    def __call__(self, *args, vm: 'VM' = None):
        """Evaluates the closure."""
        if vm is None:
            vm = self.vm
        return vm.evaluate(self.graph, args, closure=self.values)


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

    def __init__(self, implementations: Mapping[Primitive, Callable]) \
            -> None:
        """Initialize the VM."""
        self.implementations = implementations

    def convert_value(self, value):
        """Translate the value to a format that the VM understands."""
        if isinstance(value, FunctionType):
            from .api import parse
            return parse(value)
        else:
            return value

    def unconvert_value(self, value):
        """Translate a VM-produced value to a user-faced format."""
        if isinstance(value, Primitive):
            return self.implementations[value]
        elif isinstance(value, Closure):
            value.vm = self
            return value
        elif isinstance(value, Graph):
            return self.make_callable(value)
        else:
            return value

    def make_callable(self, g):
        """Return an object that executes `g` when called on arguments."""
        c = Closure(g, None)
        c.vm = self
        return c

    def evaluate(self, graph: Graph, _args: Iterable[Any], *,
                 closure: Mapping[ANFNode, Any] = None) -> Any:
        """Run a graph.

        This will evaluate the passed-in graph and return the
        resulting value.
        """
        args = smap(self.convert_value, tuple(_args))

        N = NestingAnalyzer(graph)
        self._vars = N.free_variables_total()

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        top_frame = VMFrame(toposort(graph.return_, self._succ_vm),
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
                    return smap(self.unconvert_value, r.value)

    def _succ_vm(self, node: ANFNode) -> Iterable[ANFNode]:
        """Follow node.incoming and free variables."""
        for i in node.inputs:
            if i.graph == node.graph or is_constant_graph(i):
                yield i
        if is_constant_graph(node):
            yield from self._vars[node.value]

    def _call(self, graph: Graph, args: List[Any]):
        clos = None
        if isinstance(graph, Closure):
            clos = graph.values
            graph = graph.graph

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        raise self._Call(VMFrame(toposort(graph.return_, self._succ_vm),
                                 dict(zip(graph.parameters, args)),
                                 closure=clos))

    def _make_closure(self, graph: Graph, frame: VMFrame) -> Closure:
        clos = dict()
        for v in self._vars[graph]:
            clos[v] = frame[v]
        return Closure(graph, clos)

    def _handle_node(self, node: ANFNode, frame: VMFrame):
        if isinstance(node, Constant):
            if frame.closure is not None and node in frame.closure:
                return

            # We only visit constant graphs
            assert is_constant_graph(node)
            g = node.value
            if len(self._vars[g]) != 0:
                frame.values[node] = self._make_closure(g, frame)
            # We don't need to do anything special for non-closures

        elif isinstance(node, Parameter):
            pass

        elif isinstance(node, Apply):
            fn = frame[node.inputs[0]]
            args = [frame[a] for a in node.inputs[1:]]
            if isinstance(fn, Primitive):
                if fn == if_:
                    if args[0]:
                        self._call(args[1], [])
                    else:
                        self._call(args[2], [])
                elif fn == return_:
                    raise self._Return(args[0])
                else:
                    frame.values[node] = self.implementations[fn](*args)
            else:
                self._call(fn, args)

        else:
            raise AssertionError("Unknown node type")  # pragma: no cover
