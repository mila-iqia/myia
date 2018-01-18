"""Debug/Testing Virtual Machine.

This VM will directly execute a graph so it should be suitable for
testing or debugging.  Don't expect stellar performance from this
implementation.
"""
from typing import List, Dict, Any, Union

from myia.anf_ir import Graph, ANFNode, Constant, Parameter, Apply
from myia.ir_utils import dfs
from myia.primops import Primitive, Add, If, Return


class VM:
    """Virtual Machine interface."""

    def evaluate(self, graph: Graph, args: List[Any]):
        """
        Run a graph.

        This will evaluate the passed-in graph and return the
        resulting value.

        """
        root_frame = VMFrame(graph, args, None)
        frame = root_frame
        while frame is not None:
            frame = frame.run()
        return root_frame.result()


class VMFrame:
    """
    An execution frame.

    This is the main class for the VM.  It does all the bookkeeping to
    handle the execution state.

    Attributes:
        graph: The currently executing graph.  This can change due to
            a tail call.
        args: Argument array for the current call.
        parent: VMFrame that we return to.
        closure: VMFrame to look up free variables in.
        values: Mapping of every node in the graph to its computed
            value.  Currently all the values are always kept live.
        todo: List of node that have to be visited.  Evaluation starts
            from the end of the list.

    """

    class Continue(Exception):
        """Continue step."""

    class Return(Exception):
        """End step."""

        def __init__(self, value: 'VMFrame' = None) -> None:
            """Which value to return."""
            self.value: VMFrame = value

    class Closure:
        """Representation of a closure."""

        def __init__(self, graph: Graph, frame: 'VMFrame') -> None:
            """Build a closure."""
            self.graph = graph
            self.frame = frame

    class Jump(ANFNode):
        """Jump Node for out of scope values."""

        def __init__(self, frame: 'VMFrame') -> None:
            """Set a jump."""
            super().__init__([], frame, None)

    def __init__(self, graph: Graph, args: List[Any], parent: 'VMFrame',
                 closure: 'VMFrame' = None) -> None:
        """Build a frame."""
        self.graph = graph
        self.args = args
        self.parent = parent
        self.closure = closure
        self.values: Dict[ANFNode, Any] = dict()
        self.todo: List[ANFNode] = [self.graph.return_]

    def run(self) -> 'VMFrame':
        """Run the graph to completion and return the result."""
        while True:
            try:
                res = self.advance()
            except StopIteration:
                # Return is handled here because the flow control is a bit
                # too complex for the parent frame to get a value back
                # from the called frame.
                if self.parent is not None:
                    self.parent.values[self.parent.todo[-1]] = self.result()
                    self.parent.todo.pop()
                return self.parent
            if res is not None:
                return res

    def tail(self) -> bool:
        """Are we in a tail call?."""
        return self.todo[-2] is self.graph.return_

    def done(self) -> bool:
        """Are we done?."""
        return len(self.todo) == 0

    def result(self):
        """Get this frame's result."""
        return self.values[self.graph.return_]

    def get_value(self, node: ANFNode):
        """Get the value for a node.

        This will check if we've already computed a value for it and
        add it to the queue otherwise. If there was no value, we stop
        the evaluation of the current node and restart the step, since
        no useful work was done.

        If the value we want is in another graph, we will resume
        execution in a different frame and jump back to this one later.
        """
        frame = self
        while frame and node.graph and node.graph is not frame.graph:
            frame = frame.closure
        if node not in frame.values:
            # we set up a fake node to jump back to this frame after
            # the value that we need has been evaluated.
            if frame is not self:
                frame.todo.append(self.Jump(self))
            frame.todo.append(node)
            if frame is not self:
                raise self.Return(frame)
            else:
                raise self.Continue()
        return frame.values[node]

    def wrap_closure(self, value):
        """Wrap graphs that are closures."""
        if isinstance(value, Graph):
            if any(n.graph and n.graph is not value
                   for n in dfs(value.return_)):
                return self.Closure(value, self)
        return value

    def do_call(self, graph: Union[Graph, 'VMFrame.Closure'], args: List[Any]):
        """Perform a call.

        This will handle tail calls and closures.

        """
        if self.tail() and self.parent is not None:
            parent = self.parent
        else:
            parent = self
        if isinstance(graph, self.Closure):
            new_frame = VMFrame(graph.graph, args, parent, graph.frame)
        else:
            new_frame = VMFrame(graph, args, parent)
        raise self.Return(new_frame)

    def advance(self) -> 'VMFrame':
        """
        Take a step.

        This will perform one "step" of execution.  This corresponds
        roughly to evaluating one node of the graph.  In the case of
        function calls, we single-step through the called function
        even though the apply is a single node in our graph.
        """
        if self.done():
            raise StopIteration()

        while True:
            node = self.todo[-1]
            assert node not in self.values

            try:
                self.eval_node(node)
            except self.Continue:
                continue
            except self.Return as e:
                return e.value

            self.todo.pop()
            return None

    def eval_node(self, node: ANFNode):
        """
        Compute the value for a node.

        This may fail and ask for a missing value to be computed or
        enter into a call.
        """
        if isinstance(node, Constant):
            self.values[node] = self.wrap_closure(node.value)
        elif isinstance(node, Parameter):
            idx = self.graph.parameters.index(node)
            self.values[node] = self.args[idx]
        elif isinstance(node, self.Jump):
            self.todo.pop()
            raise self.Return(node.value)
        elif isinstance(node, Apply):
            fn = self.get_value(node.inputs[0])
            args = [self.get_value(i) for i in node.inputs[1:]]
            if isinstance(fn, Primitive):
                if isinstance(fn, Add):
                    self.values[node] = sum(args[1:], args[0])
                elif isinstance(fn, Return):
                    self.values[node] = args[0]
                elif isinstance(fn, If):
                    cond = args[0]
                    if cond:
                        inner = args[1]
                    else:
                        inner = args[2]
                    self.do_call(inner, [])
                else:
                    raise ValueError('Unknown primitive')  # pragma: no cover
            else:
                self.do_call(fn, args)
