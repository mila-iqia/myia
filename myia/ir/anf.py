"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""

from collections import defaultdict
from typing import Any, Iterable, List, Union, Dict

from ..dtype import Function
from ..info import NamedDebugInfo
from ..prim import ops as primops, Primitive
from ..utils import Named, list_str, repr_, UNKNOWN
from ..utils.unify import expandlist, noseq

from .abstract import Node

PARAMETER = Named('PARAMETER')
SPECIAL = Named('SPECIAL')
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
        debug: A NamedDebugInfo object containing debugging information about
            this graph.
        transforms: A dictionary of available transforms for this graph, e.g.
            'grad' or 'primal'.

    """

    def __init__(self) -> None:
        """Construct a graph."""
        self.parameters: List[Parameter] = []
        self.return_: Apply = None
        self.debug = NamedDebugInfo(self)
        self.flags = {}
        self.transforms: Dict[str, Union[Graph, Primitive]] = {}
        self._manager = None

    @property
    def type(self):
        """Return the graph's type based on parameter/output types."""
        if any(p.type is UNKNOWN for p in self.parameters):
            return UNKNOWN
        return Function[tuple(p.type for p in self.parameters),
                        self.output.type]

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
            if self._manager:
                self._manager.set_edge(self.return_, 1, value)
            else:
                self.return_.inputs[1] = value
        else:
            self.return_ = Apply([Constant(primops.return_), value], self)
        self.return_.type = value.type
        if value.type is not UNKNOWN:
            self.return_.inputs[0].type = Function[(value.type,), value.type]

    def add_parameter(self) -> 'Parameter':
        """Add a new parameter to this graph (appended to the end)."""
        p = Parameter(self)
        new_parameters = self.parameters + [p]
        if self._manager is None:
            self.parameters = new_parameters
        else:
            self._manager.set_parameters(self, new_parameters)
        return p

    def constant(self, obj: Any) -> 'Constant':
        """Create a constant for the given object."""
        return Constant(obj)

    def apply(self, *inputs: Any) -> 'Apply':
        """Create an Apply node with given inputs, bound to this graph."""
        wrapped_inputs = [i if isinstance(i, ANFNode) else self.constant(i)
                          for i in inputs]
        return Apply(wrapped_inputs, self)

    ######################
    # Managed properties #
    ######################

    @property
    def manager(self):
        """Return the GraphManager for this Graph."""
        if self._manager is None:
            raise Exception(f'Graph {self} has no manager.')
        return self._manager

    @property
    def nodes(self):
        """Return all nodes that belong to this graph."""
        return self.manager.nodes[self]

    @property
    def constants(self):
        """Return all constants used by this graph."""
        return self.manager.constants[self]

    @property
    def free_variables_direct(self):
        """Return all free variables directly pointed to in this graph."""
        return self.manager.free_variables_direct[self]

    @property
    def free_variables_total(self):
        """Return all free variables required by this graph's scope."""
        return self.manager.free_variables_total[self]

    @property
    def graphs_used(self):
        """Return all graphs used by this graph directly."""
        return self.manager.graphs_used[self]

    @property
    def graph_users(self):
        """Return all graphs that use this graph."""
        return self.manager.graph_users[self]

    @property
    def graph_dependencies_direct(self):
        """Return the set of graphs free_variables_direct belong to."""
        return self.manager.graph_dependencies_direct[self]

    @property
    def graph_dependencies_total(self):
        """Return the set of graphs free_variables_total belong to."""
        return self.manager.graph_dependencies_total[self]

    @property
    def parent(self):
        """Return the parent of this graph."""
        return self.manager.parents.get(self, None)

    @property
    def children(self):
        """Return all graphs that have this graph as parent."""
        return self.manager.children[self]

    @property
    def scope(self):
        """Return this graph and all nested graphs."""
        return self.manager.scopes[self]

    @property
    def graphs_reachable(self):
        """Return all graphs that may figure this one's call tree."""
        return self.manager.graphs_reachable[self]

    @property
    def recursive(self):
        """Return whether this graph is recursive."""
        return self.manager.recursive[self]

    #################
    # Miscellaneous #
    #################

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
        self.inputs = list(inputs)
        self.value = value
        self.graph = graph
        self.debug = NamedDebugInfo(self)
        self.inferred = defaultdict(lambda: UNKNOWN)

    @property
    def type(self):
        """Return the node's type."""
        return self.inferred['type']

    @type.setter
    def type(self, value):
        """Set the node's type."""
        self.inferred['type'] = value

    @property
    def incoming(self) -> Iterable['ANFNode']:
        """Return incoming nodes in order."""
        return iter(self.inputs)

    def __str__(self) -> str:
        return self.debug.debug_name

    ##########
    # Checks #
    ##########

    def is_apply(self, value: Any = None) -> bool:
        """Return whether self is an Apply."""
        return False

    def is_parameter(self) -> bool:
        """Return whether self is a Parameter."""
        return False

    def is_constant(self, cls: Any = object) -> bool:
        """Return whether self is a Constant, with value of given cls."""
        return False

    def is_constant_graph(self) -> bool:
        """Return whether self is a Constant with a Graph value."""
        return False

    def is_special(self, cls: Any = object) -> bool:
        """Return whether self is a Special, with value of given cls."""
        return False


class Apply(ANFNode):
    """A function application.

    This node represents the application of a function to a set of arguments.

    """

    def __init__(self, inputs: List[ANFNode], graph: 'Graph') -> None:
        """Construct an application."""
        super().__init__(inputs, APPLY, graph)

    def is_apply(self, value: Any = None) -> bool:
        """Return whether self is an Apply."""
        if value is not None:
            fn = self.inputs[0]
            return fn.is_constant() and fn.value is value
        else:
            return True

    def __visit__(self, fn):
        new_inputs = expandlist(map(fn, self.inputs))
        g = noseq(fn, self.graph)
        app = Apply(new_inputs, g)
        app.type = self.type
        return app

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

    def is_parameter(self):
        """Return whether self is a Parameter."""
        return True

    def __visit__(self, fn):
        g = noseq(fn, self.graph)
        if not isinstance(g, Graph) or g is not self.graph:
            # Note: this condition will be triggered if e.g. there is a
            # Parameter in a pattern to reify. It's not clear what that's
            # supposed to mean unless the Parameter already exists in a
            # concrete graph, so we raise an Exception just in case.
            raise Exception('Unification cannot create new Parameters.') \
                # pragma: no cover
        return self

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

    def is_constant(self, cls: Any = object) -> bool:
        """Return whether self is a Constant, with value of given cls."""
        return isinstance(self.value, cls)

    def is_constant_graph(self) -> bool:
        """Return whether self is a Constant with a Graph value."""
        return self.is_constant(Graph)

    def __visit__(self, fn):
        ct = Constant(noseq(fn, self.value))
        ct.type = self.type
        return ct

    def __str__(self) -> str:
        if isinstance(self.value, LITERALS):
            return str(self.value)
        return super().__str__()

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, value=self.value)


class Special(ANFNode):
    """A special node.

    This is generally not a legal node in a graph, but may be needed by special
    purpose algorithms, e.g. to hold a Var when performing unification on
    graphs.

    Attributes:
        special: Some object that this node is wrapping.

    """

    def __init__(self, special: Any, graph: Graph) -> None:
        """Initialize a special node."""
        super().__init__([], SPECIAL, graph)
        self.special = special

    def is_special(self, cls: Any = object) -> bool:
        """Return whether self is a Special, with value of given cls."""
        return isinstance(self.special, cls)

    def __str__(self) -> str:
        return str(self.special)  # pragma: no cover

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, special=self.special) \
            # pragma: no cover
