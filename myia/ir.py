"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""
from typing import (List, Set, Tuple, Dict, Any, Optional, Sequence,
                    MutableSequence, overload, Iterable)

from myia.utils import Named

PARAMETER = Named('PARAMETER')
RETURN = Named('RETURN')


class Graph:
    """A function graph.

    Attributes:
        parameters: The parameters of this function as a list of `Parameter`
            nodes. Parameter nodes that are unreachable by walking from the
            output node correspond to unused parameters.
        return_: The `Return` node whose value will be returned by this
            function. A graph initially has no output node (because it won't be
            known e.g. until the function has completed parsing), but it must
            be set afterwards for the graph instance to be valid.

    """

    def __init__(self) -> None:
        """Construct a graph."""
        self.parameters: List[Parameter] = []
        self.return_: Optional[Return] = None
        self.debug: Dict = {}


class Node:
    """A node in the graph-based ANF IR.

    There are four types of nodes: Function applications; parameters; return
    values; and constants such as numbers and functions.

    Attributes:
        inputs: If the node is a function application, the first node input is
            the function to apply, followed by the arguments. These are use-def
            edges.
        value: The value of this node, if it is a constant. Parameters have the
            special value `PARAMETER`.
        graph: The function definition graph that this node belongs to for
            values and parameters.
        uses: A set of tuples with the nodes that use this node alongside with
            the index. These def-use edges are the reverse of the `inputs`
            attribute, creating a doubly linked graph structure.
        debug: A dictionary with debug information about this node e.g. a
            human-readable name and the Python source code.

    """

    def __init__(self, inputs: List['Node'], value: Any,
                 graph: Optional[Graph]) -> None:
        """Construct a node."""
        self.inputs = Inputs(self, inputs)
        self.value = value
        self.graph = graph
        self.uses: Set[Tuple[Node, int]] = set()
        self.debug: Dict = {}


class Inputs(MutableSequence[Node]):
    """Container data structure for node inputs.

    This mutable sequence data structure can be used to keep track of a node's
    inputs. Any insertion or deletion of edges will be reflected in the inputs'
    `uses` attribute.

    """

    def __init__(self, node: Node, initlist: Iterable[Node] = None) -> None:
        """Construct the inputs container for a node.

        Args:
            node: The node of which the inputs are stored.
            initlist: A sequence of nodes to initialize the container with.

        """
        self.node = node
        self.data: List[Node] = []
        if initlist is not None:
            self.extend(initlist)

    @overload
    def __getitem__(self, index: int) -> Node:
        pass

    @overload  # noqa: F811
    def __getitem__(self, s: slice) -> Sequence[Node]:
        pass

    def __getitem__(self, index):  # noqa: F811
        """Get an input by its index."""
        return self.data.__getitem__(index)

    @overload
    def __setitem__(self, index: int, value: Node) -> None:
        pass

    @overload  # noqa: F811
    def __setitem__(self, index: slice, value: Iterable[Node]) -> None:
        pass

    def __setitem__(self, index, value):  # noqa: F811
        """Replace an input with another."""
        if isinstance(index, slice):
            for i, v in zip(range(*index.indices(len(self))), value):
                self[i] = v
        old_value = self.data[index]
        if old_value.uses is not None:
            old_value.uses.remove((self.node, index))
        if value.uses is not None:
            value.uses.add((self.node, index))
        self.data.__setitem__(index, value)

    @overload
    def __delitem__(self, index: int) -> None:
        pass

    @overload  # noqa: F811
    def __delitem__(self, index: slice) -> None:
        pass

    def __delitem__(self, index):  # noqa: F811
        """Delete an input."""
        if isinstance(index, slice):
            for i in range(*index.indices(len(self))):
                del self[i]
        value = self.data[index]
        if value.uses is not None:
            value.uses.remove((self.node, index))
        self.data.__delitem__(index)

    def __len__(self) -> int:
        """Get the number of inputs."""
        return len(self.data)

    def insert(self, index: int, value: Node) -> None:
        """Insert an input at a given location."""
        for i, next_value in enumerate(reversed(self.data[index:])):
            if next_value.uses is not None:
                next_value.uses.remove((self.node, len(self) - i - 1))
                next_value.uses.add((self.node, len(self) - i))
        if value.uses is not None:
            value.uses.add((self.node, index))
        self.data.insert(index, value)

    def __repr__(self) -> str:
        """Return a string representation of the inputs."""
        return f"Inputs({self.data})"


class Apply(Node):
    """A function application.

    This node represents the application of a function to a set of arguments.

    """

    def __init__(self, inputs: List[Node], graph: 'Graph') -> None:
        """Construct an application."""
        if len(inputs) < 1:
            raise ValueError("at least one input must be provided")
        super().__init__(inputs, None, graph)


class Parameter(Node):
    """A parameter to a function.

    Parameters are leaf nodes, since they are not the result of a function
    application, and they have no value. They are entirely defined by the graph
    they belong to.

    """

    def __init__(self, graph: Graph) -> None:
        """Construct the parameter."""
        super().__init__([], PARAMETER, graph)


class Return(Node):
    """The value returned by a function.

    Return nodes have exactly one input, which points to the value that the
    function will return. They are a root node in the function graph.

    """

    def __init__(self, input_: Node, graph: Graph) -> None:
        """Construct a return node."""
        super().__init__([input_], RETURN, graph)


class Constant(Node):
    """A constant node.

    A constant is a node which is not the result of a function application. In
    the graph it is a leaf node. It has no inputs, and instead is defined
    entirely by its value. Unlike parameters and values, constants do not
    belong to any particular function graph.

    Two "special" constants are those whose value is a `Primitive`
    (representing primitive operations) or whose value is a `Graph` instance
    (representing functions).

    """

    def __init__(self, value: Any) -> None:
        """Construct a literal."""
        super().__init__([], value, None)
