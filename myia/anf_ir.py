"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""
from typing import (List, Set, Tuple, Any, Sequence, MutableSequence,
                    overload, Iterable)

from myia.ir import Node
from myia.utils import Named, repr_, list_str
from myia.info import NamedDebugInfo
from myia import primops

PARAMETER = Named('PARAMETER')
APPLY = Named('APPLY')

LITERALS = (bool, int, str, float)


class Graph:
    """A function graph.

    Attributes:
        parameters: The parameters of this function as a list of `Parameter`
            nodes. Parameter nodes that are unreachable by walking from the
            output node correspond to unused parameters.
        return_: The `Apply` node that calls the `Return` primitive. The input
            to this node will be returned by this function. A graph initially
            has no output node (because it won't be known e.g. until the
            function has completed parsing), but it must be set afterwards for
            the graph instance to be valid.

    """

    def __init__(self) -> None:
        """Construct a graph."""
        self.parameters: List[Parameter] = []
        self.return_: Apply = None
        self.debug = NamedDebugInfo(self)

    @property
    def output(self) -> 'ANFNode':
        """
        Return the graph's output.

        Equal to `self.return_.inputs[1]`, if it exists. Unlike `return_`,
        `output' may be a constant or belong to a different graph.
        """
        if not self.return_ or len(self.return_.inputs) != 2:
            raise Exception('Graph has no output.')
        return self.return_.inputs[1]

    @output.setter
    def output(self, value: 'ANFNode') -> None:
        """Set the graph's output."""
        if self.return_:
            self.return_.inputs[1] = value
        else:
            self.return_ = Apply([Constant(primops.return_), value], self)

    def __str__(self) -> str:
        return self.debug.debug_name

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name,
                     parameters=list_str(self.parameters),
                     return_=self.return_)


class ANFNode(Node):
    """A node in the graph-based ANF IR.

    There are three types of nodes: Function applications; parameters; and
    constants such as numbers and functions.

    Attributes:
        inputs: If the node is a function application, the first node input is
            the function to apply, followed by the arguments. These are use-def
            edges. For other nodes, this attribute is empty.
        value: The value of this node if it is a constant. Parameters and
            function applications have the special values `PARAMETER` and
            `APPLY`.
        graph: The function definition graph that this node belongs to for
            values and parameters (constants don't belong to any function).
        uses: A set of tuples with the nodes that use this node alongside with
            the index. These def-use edges are the reverse of the `inputs`
            attribute, creating a doubly linked graph structure. Note that this
            container is updated automatically; do not manipulate it manually.
        debug: An object with debug information about this node e.g. a
            human-readable name and the Python source code.

    """

    def __init__(self, inputs: Iterable['ANFNode'], value: Any,
                 graph: Graph) -> None:
        """Construct a node."""
        self._inputs = Inputs(self, inputs)
        self.value = value
        self.graph = graph
        self.uses: Set[Tuple[ANFNode, int]] = set()
        self.debug = NamedDebugInfo(self)

    @property
    def inputs(self) -> 'Inputs':
        """Return the list of inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, value: Iterable['ANFNode']) -> None:
        """Set the list of inputs."""
        self._inputs.clear()  # type: ignore
        self._inputs = Inputs(self, value)

    @property
    def incoming(self) -> Iterable['ANFNode']:
        """Return incoming nodes in order."""
        return iter(self.inputs)

    @property
    def outgoing(self) -> Iterable['ANFNode']:
        """Return uses of this node in random order."""
        return (node for node, index in self.uses)

    def __copy__(self) -> 'ANFNode':
        """Copy this node.

        This method is used by the `copy` module. It ensures that copied nodes
        will have correct `uses` information. Debug information is not copied.

        """
        cls = self.__class__
        obj = cls.__new__(cls)  # type: ignore
        ANFNode.__init__(obj, self.inputs, self.value, self.graph)
        return obj

    def __str__(self) -> str:
        return self.debug.debug_name


class Inputs(MutableSequence[ANFNode]):
    """Container data structure for node inputs.

    This mutable sequence data structure can be used to keep track of a node's
    inputs. Any insertion or deletion of edges will be reflected in the inputs'
    `uses` attributes.

    """

    def __init__(self, node: ANFNode,
                 initlist: Iterable[ANFNode] = None) -> None:
        """Construct the inputs container for a node.

        Args:
            node: The node of which the inputs are stored.
            initlist: A sequence of nodes to initialize the container with.

        """
        self.node = node
        self.data: List[ANFNode] = []
        if initlist is not None:
            self.extend(initlist)

    @overload
    def __getitem__(self, index: int) -> ANFNode:
        pass

    @overload  # noqa: F811
    def __getitem__(self, index: slice) -> Sequence[ANFNode]:
        pass

    def __getitem__(self, index):  # noqa: F811
        """Get an input by its index."""
        return self.data[index]

    @overload
    def __setitem__(self, index: int, value: ANFNode) -> None:
        pass

    @overload  # noqa: F811
    def __setitem__(self, index: slice, value: Iterable[ANFNode]) -> None:
        pass

    def __setitem__(self, index, value):  # noqa: F811
        """Replace an input with another."""
        if isinstance(index, slice):
            raise ValueError("slice assignment not supported")
        if index < 0:
            index += len(self)
        old_value = self.data[index]
        old_value.uses.remove((self.node, index))
        value.uses.add((self.node, index))
        self.data[index] = value

    @overload
    def __delitem__(self, index: int) -> None:
        pass

    @overload  # noqa: F811
    def __delitem__(self, index: slice) -> None:
        pass

    def __delitem__(self, index):  # noqa: F811
        """Delete an input."""
        if isinstance(index, slice):
            raise ValueError("slice deletion not supported")
        if index < 0:
            index += len(self)
        value = self.data[index]
        value.uses.remove((self.node, index))
        for i, next_value in enumerate(self.data[index + 1:]):
            next_value.uses.remove((self.node, i + index + 1))
            next_value.uses.add((self.node, i + index))
        del self.data[index]

    def __len__(self) -> int:
        """Get the number of inputs."""
        return len(self.data)

    def insert(self, index: int, value: ANFNode) -> None:
        """Insert an input at a given location."""
        if index < 0:
            index += len(self)
        for i, next_value in enumerate(reversed(self.data[index:])):
            next_value.uses.remove((self.node, len(self) - i - 1))
            next_value.uses.add((self.node, len(self) - i))
        value.uses.add((self.node, index))
        self.data.insert(index, value)

    def __str__(self) -> str:
        return list_str(self.data)

    def __repr__(self) -> str:
        return f"Inputs({self.node}, {list_str(self.data)})"

    def __eq__(self, other) -> bool:
        """Test whether a list of inputs is equal to another list.

        Note:
            The goal of `Inputs` is to behave exactly like a list, but track
            insertions and deletions to keep the bidirectional graph structure
            up to date. As such, which node an `Inputs` object is attached to
            is irrelevant to the equality test. Instead, a simple element by
            element test is performed.

        """
        return all(x == y for x, y in zip(self, other))


class Apply(ANFNode):
    """A function application.

    This node represents the application of a function to a set of arguments.

    """

    def __init__(self, inputs: List[ANFNode], graph: 'Graph') -> None:
        """Construct an application."""
        super().__init__(inputs, APPLY, graph)

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, inputs=self.inputs,
                     graph=self.graph)


class Parameter(ANFNode):
    """A parameter to a function.

    Parameters are leaf nodes, since they are not the result of a function
    application, and they have no value. They are entirely defined by the graph
    they belong to.

    """

    def __init__(self, graph: Graph) -> None:
        """Construct the parameter."""
        super().__init__([], PARAMETER, graph)

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, graph=self.graph)


class Constant(ANFNode):
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

    def __str__(self) -> str:
        if isinstance(self.value, LITERALS):
            return str(self.value)
        return super().__str__()

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, value=self.value)
