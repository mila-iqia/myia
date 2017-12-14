"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""
from enum import Enum, auto
from typing import List, Dict, Any, Optional

from myia.utils import Named

PARAMETER = Named('PARAMETER')


class Graph:
    """A function graph.

    Attributes:
        output: The `Node` whose value will be returned by this function.

    """

    def __init__(self, output: 'Node') -> None:
        """Construct a graph."""
        self.output = output
        self.debug: Dict = {}


class Node:
    """A node in the graph-based ANF IR.

    There are three types of nodes: Function applications; parameters; and
    constants such as numbers and functions.

    Attributes:
        inputs: If the node is a function application, the first node input is
            the function to apply, followed by the arguments.
        uses: A list of function applications that use this node as a function
            or argument. This attribute is optional and initially set to
            `None`.
        value: The value of this node, if it is a constant. Parameters have the
            special value `PARAMETER`.
        graph: The function definition graph that this node belongs to for
            values and parameters.
        debug: A dictionary with debug information about this node e.g. a
            human-readable name and the Python source code.

    """

    def __init__(self) -> None:
        """Construct a node."""
        self.inputs: List[Node] = []
        self.uses: Optional[List[Node]] = None
        self.value: Any = None
        self.graph: Graph = None
        self.debug: Dict = {}


class Apply(Node):
    """A function application.

    This node represents the application of a function to a set of arguments.

    """

    def __init__(self, inputs: List[Node], graph: 'Graph') -> None:
        """Construct an application."""
        if len(inputs) < 1:
            raise ValueError("at least one input must be provided")
        super().__init__()
        self.inputs = inputs
        self.graph = graph


class Parameter(Node):
    """A parameter to a function.

    Parameters are leaf nodes, since they are not the result of a function
    application, and they have no value. They are entirely defined by the graph
    they belong to.

    """

    def __init__(self, graph: Graph) -> None:
        """Construct the parameter."""
        super().__init__()
        self.value = PARAMETER
        self.graph = graph


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
        super().__init__()
        self.value = value


class Primitive(Enum):
    """Built-in primitive operations."""

    ADD = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()
