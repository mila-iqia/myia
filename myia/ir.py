"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""
from typing import List, Set, Tuple, Dict, Any, Optional

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
            attribute, creating a doubly linked graph structure. This attribute
            is optional and initially set to `None`. If this attribute is
            present, care must be taken to make sure that changes to `uses` is
            reflected in the other nodes' `inputs` attribute and vice versa.
        debug: A dictionary with debug information about this node e.g. a
            human-readable name and the Python source code.

    """

    def __init__(self, inputs: List['Node'], value: Any,
                 graph: Optional[Graph]) -> None:
        """Construct a node."""
        self.inputs = inputs
        self.value = value
        self.graph = graph
        self.uses: Optional[Set[Tuple[Node, int]]] = None
        self.debug: Dict = {}


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
