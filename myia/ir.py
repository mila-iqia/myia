"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""
from enum import Enum, auto
from typing import List, Union, Dict

LiteralType = Union[str, int, float, 'PrimitiveOperation', 'Graph', None]


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

    There are three types of nodes: Values, which are the result of function
    applications; parameters; and constants such as numbers and functions.

    Attributes:
        inputs: If the node is a value, the first node input is the function to
            apply, followed by the arguments.
        value: The value of this node, if it is a constant.
        graph: The function definition graph that this node belongs to for
            values and parameters.
        debug: A dictionary with debug information about this node e.g. a
            human-readable name and the Python source code.

    """

    def __init__(self) -> None:
        """Construct a node."""
        self.inputs: List[Node] = []
        self.value: LiteralType = None
        self.graph: Graph = None
        self.debug: Dict = {}


class Value(Node):
    """A value.

    A value is the result of function application.

    """

    def __init__(self, inputs: List[Node], graph: 'Graph') -> None:
        """Construct a value."""
        if len(inputs) < 1:
            raise ValueError
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
        self.graph = graph


class Constant(Node):
    """A constant node.

    A constant is a node which is not the result of a function application. In
    the graph it is a leaf node. It has no inputs, and instead is defined
    entirely by its value. Unlike parameters and values, constants do not
    belong to any particular function graph.

    """

    def __init__(self, value: LiteralType) -> None:
        """Construct a literal."""
        super().__init__()
        self.value = value


class Literal(Constant):
    """A literal.

    A literal is a string or a numeric value.

    """

    def __init__(self, value: Union[str, float, int]) -> None:
        """Construct a literal."""
        super().__init__(value)


class PrimitiveOperation(Enum):
    """Built-in primitive operations."""

    ADD = auto()
    SUB = auto()
    MULT = auto()
    DIV = auto()


class Primitive(Constant):
    """A primitive.

    A primitive operation is a built-in such as addition and multiplication.

    """

    def __init__(self, value: PrimitiveOperation) -> None:
        """Construct the primitive."""
        super().__init__(value)


class Function(Constant):
    """A user-defined function."""

    def __init__(self, value: Graph) -> None:
        """Construct the function."""
        super().__init__(value)
