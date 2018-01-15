"""Debug/Testing Virtual Machine.

This VM will directly execute a graph so it should be suitable for
testing or debugging.  Don't expect stellar performance from this
implementation.
"""
from typing import List, Dict, Any

from myia.anf_ir import Graph, ANFNode, Constant, Parameter, Apply
from myia.primops import Primitive, Add, If, Return


class VM:
    """
    Virtual Machine interface.
    """

    def evaluate(self, graph: Graph, args: List[Any]):
        """
        Run a graph.

        This will evaluate the passed-in graph and return the
        resulting value.

        """
        frame = VMFrame(graph, args)
        return frame.evaluate()


class VMFrame:
    """
    An execution frame.

    This is the main class for the VM.  It does all the bookkeeping to
    handle the execution state.

    Attributes:
        graph: The currently executing graph.  This can change due to
            a tail call.
        args: Argument array for the current call.
        values: Mapping of every node in the graph to its computed
            value.  Currently all the values are always kept live.
        todo: List of node that have to be visited.  Evaluation starts
            from the end of the list.

    """

    class Continue(Exception):
        """Continue step."""

    class Return(Exception):
        """End step."""

    def __init__(self, graph: Graph, args: List[Any]) -> None:
        """Build a frame."""
        self.values: Dict[ANFNode, Any] = dict()
        self.reset(graph, args)
        self.calling: VMFrame = None

    def reset(self, graph: Graph, args: List[Any]) -> None:
        """
        Set up a call.

        This is used while the frame is executing a graph to implement
        tail calls.
        """
        self.graph = graph
        self.args = args
        self.todo: List[ANFNode] = [self.graph.return_]

    def evaluate(self):
        """Run the graph to completion and return the result."""
        while True:
            try:
                self.advance()
            except StopIteration:
                return self.values[self.graph.return_]

    def tail(self):
        """Are we in a tail call?."""
        return self.todo[-2] is self.graph.return_

    def done(self):
        """Are we done?."""
        return self.graph.return_ in self.values

    def get_value(self, node: ANFNode):
        """
        Get the value for a node.

        This will check if we've already computed a value for it and
        add it to the queue otherwise. If there was no value, we stop
        the evaluation of the current node and restart the step, since
        no useful work was done.
        """
        if node not in self.values:
            self.todo.append(node)
            raise self.Continue()
        return self.values[node]

    def do_call(self, graph: Graph, args: List[Any]):
        """
        Perform a call.

        This will handle tail calls correctly by reusing the current VMFrame.
        """
        if self.tail():
            self.reset(graph, args)
            raise self.Return()
        self.calling = VMFrame(graph, args)

    def advance(self):
        """
        Take a step.

        This will perform one "step" of execution.  This corresponds
        roughly to evaluating one node of the graph.  In the case of
        function calls, we single-step through the called function
        even though the apply is a single node in our graph.
        """
        if self.done():
            raise StopIteration()

        # This hackish section is to handle single-stepping through calls.
        if self.calling:
            try:
                self.calling.advance()
            except StopIteration:
                retval = self.calling.values[self.graph.return_]
                self.values[self.todo[-1]] = retval
                self.todo.pop()
                self.calling = None
                return

        while True:
            node = self.todo[-1]
            # Skip nodes that have already been evaluated
            if node in self.values:
                self.todo.pop()
                continue

            try:
                self.eval_node(node)
            except self.Continue:
                continue
            except self.Return:
                return

            self.todo.pop()
            return

    def eval_node(self, node: ANFNode):
        """
        Compute the value for a node.

        This may fail and ask for a missing value to be computed or
        enter into a call.
        """
        if isinstance(node, Constant):
            self.values[node] = node.value
        elif isinstance(node, Parameter):
            idx = self.graph.parameters.index(node)
            self.values[node] = self.args[idx]
        elif isinstance(node, Apply):
            fn = self.get_value(node.inputs[0])
            if isinstance(fn, Primitive):
                if isinstance(fn, Add):
                    args = [self.get_value(i) for i in node.inputs[1:]]
                    self.values[node] = sum(args[1:], args[0])
                elif isinstance(fn, Return):
                    self.values[node] = self.get_value(node.inputs[1])
                elif isinstance(fn, If):
                    cond = self.get_value(node.inputs[1])
                    if cond:
                        inner = self.get_value(node.inputs[2])
                    else:
                        inner = self.get_value(node.inputs[3])
                    self.do_call(inner, [])
                else:
                    raise ValueError('Unknown primitive')
            else:
                args = [self.get_value(i) for i in node.inputs[1:]]
                self.do_call(fn, args)
