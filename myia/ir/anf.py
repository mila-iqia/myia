"""Intermediate representation definition.

Myia's main intermediate representation (IR) is a graph-based version of ANF.
Each function definition (lambda) is defined as a graph, consisting of a series
of function applications.

A function can be applied to a node from another funtion's graph; this
implicitly creates a nested function. Functions are first-class objects, so
returning a nested function creates a closure.

"""

from copy import copy
from typing import Any, Dict, Iterable, List, Union

from ..info import About, NamedDebugInfo
from ..operations import Primitive, primitives as primops
from ..utils import Named, list_str, repr_, serializable
from ..utils.unify import Unification, expandlist, noseq
from .abstract import Node

PARAMETER = Named('PARAMETER')
SPECIAL = Named('SPECIAL')
APPLY = Named('APPLY')


@serializable('Graph')
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
        vararg (bool): Whether there is an *args argument. This will be the
            last parameter unless there is a kwarg, in which case it will be
            the second-to-last.
        kwarg (bool): Whether there is a *kwargs argument. This will always be
            the last parameter.
        defaults: List of parameter names that have default values.
        kwonly: The number of keyword-only arguments, which are all at the end
            of the parameters list or immediately before vararg and/or kwarg.
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
        self.vararg = False
        self.kwarg = False
        self.defaults = []
        self.kwonly = 0
        self._user_graph = None
        self._sig = None
        self._manager = None

    def plain(self):
        """Return whether the graph is plain, i.e. not parameterized."""
        return (not self.vararg
                and not self.kwarg
                and self.defaults == []
                and self.kwonly == 0)

    def _serialize(self):
        assert self.plain()
        return {'parameters': self.parameters,
                'return': self.return_,
                'debug': self.debug}

    @classmethod
    def _construct(cls):
        g = cls()
        data = yield g
        g.parameters = data['parameters']
        g.return_ = data['return']
        g.debug = data['debug']

    @property
    def abstract(self):
        """Return the graph's type based on parameter/output types."""
        from ..abstract import VirtualFunction, AbstractFunction
        if any(p.abstract is None for p in self.parameters):
            return None  # pragma: no cover
        vf = VirtualFunction(tuple(p.abstract for p in self.parameters),
                             self.output.abstract)
        return AbstractFunction(vf)

    @property
    def output(self) -> 'ANFNode':
        """
        Return the graph's output.

        Equal to `self.return_.inputs[1]`, if it exists. Unlike `return_`,
        `output' may be a constant or belong to a different graph.
        """
        if not self.return_ or len(self.return_.inputs) < 2:
            raise Exception('Graph has no output.')
        return self.return_.inputs[1]

    @output.setter
    def output(self, value: 'ANFNode') -> None:
        """Set the graph's output."""
        from ..abstract import AbstractFunction, PrimitiveFunction
        if self.return_:
            if self._manager:
                self._manager.set_edge(self.return_, 1, value)
            else:
                self.return_.inputs[1] = value
        else:
            self.return_ = Apply([Constant(primops.return_), value], self)
        self.return_.abstract = value.abstract
        f = PrimitiveFunction(primops.return_,
                              tracking_id=self.return_.inputs[0])
        self.return_.inputs[0].abstract = AbstractFunction(f)

    @property
    def parameter_names(self):
        """Return a list of parameter names."""
        from ..debug.label import label
        return [label(p) for p in self.parameters]

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

    def make_new(self, relation='copy'):
        """Make a new graph that's about this one."""
        with About(self.debug, relation):
            g = type(self)()
        g.flags = copy(self.flags)
        g.transforms = copy(self.transforms)
        g.vararg = self.vararg
        g.kwarg = self.kwarg
        g.defaults = self.defaults
        g.kwonly = self.kwonly
        g._user_graph = self._user_graph
        g._sig = self._sig
        return g

    #######################
    # MetaGraph interface #
    #######################

    async def normalize_args(self, args):
        """Return normalized versions of the arguments.

        By default, this returns args unchanged.
        """
        return self.normalize_args_sync(args)

    def normalize_args_sync(self, args):
        """Return normalized versions of the arguments.

        By default, this returns args unchanged.
        """
        return args

    def make_signature(self, args):
        """Return a signature corresponding to the args.

        Each signature corresponds to a graph.
        """
        from ..abstract.data import AbstractKeywordArgument
        if args is None:
            return None
        nargs = 0
        keys = []
        for arg in args:
            if isinstance(arg, AbstractKeywordArgument):
                keys.append(arg.key)
            else:
                assert not keys
                nargs += 1
        return (nargs, *keys)

    def generate_graph(self, sig):
        """Generate a Graph for the given abstract arguments."""
        from .clone import clone
        from .manager import GraphManager
        from ..abstract import AbstractDict, ANYTHING
        from ..utils import MyiaTypeError
        from ..operations import primitives as P

        if sig is None:
            return self

        if sig == self._sig:
            return self

        if self._user_graph:
            return self._user_graph.generate_graph(sig)

        nargs, *keys = sig
        if (not self.defaults and not self.vararg and not self.kwarg
                and not self.kwonly and not keys):
            return self

        new_graph = clone(self, total=True)
        repl = {}

        max_n_pos = (len(self.parameters)
                     - bool(self.vararg)
                     - bool(self.kwarg)
                     - self.kwonly)
        n_pos = min(max_n_pos, nargs)

        new_order = new_graph.parameters[:n_pos]

        n_var = nargs - n_pos

        vararg = None
        if self.vararg:
            vararg = new_graph.parameters[-1 - bool(self.kwarg)]
            v_parameters = []
            for i in range(n_var):
                new_param = Parameter(new_graph)
                new_param.debug.name = f'{self.vararg}[{i}]'
                v_parameters.append(new_param)
            new_order += v_parameters
            constructed = new_graph.apply(
                P.make_tuple, *v_parameters
            )
            repl[vararg] = constructed
        elif n_var:
            raise MyiaTypeError(f'Too many arguments')

        if self.kwarg:
            kwarg = new_graph.parameters[-1]
        else:
            kwarg = None

        kwarg_parts = []
        kwarg_keys = []
        if keys:
            for k in keys:
                try:
                    idx = self.parameter_names.index(k)
                except ValueError:
                    if kwarg:
                        new_param = Parameter(new_graph)
                        new_param.debug.name = f'{self.kwarg}[{k}]'
                        kwarg_parts.append(
                            new_graph.apply(P.extract_kwarg, k, new_param)
                        )
                        kwarg_keys.append(k)
                        new_order.append(new_param)
                    else:
                        raise MyiaTypeError(f'Invalid keyword argument: {k}')
                else:
                    p = new_graph.parameters[idx]
                    if p in new_order:
                        raise MyiaTypeError(
                            f'Multiple values given for argument {k}'
                        )
                    new_order.append(p)
                    repl[p] = new_graph.apply(P.extract_kwarg, k, p)

        if kwarg:
            typ = AbstractDict(dict((key, ANYTHING) for key in kwarg_keys))
            repl[kwarg] = new_graph.apply(P.make_dict, typ, *kwarg_parts)

        all_defaults = new_graph.return_.inputs[2:]
        for name, param in zip(new_graph.parameter_names,
                               new_graph.parameters):
            if param not in (vararg, kwarg) and param not in new_order:
                try:
                    idx = self.defaults.index(name)
                except ValueError:
                    raise MyiaTypeError(f'Missing argument: {name}')
                repl[param] = all_defaults[idx]

        mng = GraphManager(manage=False, allow_changes=True)
        mng.add_graph(new_graph)
        with mng.transact() as tr:
            for x, y in repl.items():
                tr.replace(x, y)
            tr.set_parameters(new_graph, new_order)

        new_graph.return_.inputs[2:] = []
        new_graph.vararg = False
        new_graph.kwarg = False
        new_graph.defaults = []
        new_graph.kwonly = 0
        new_graph._user_graph = self
        new_graph._sig = sig
        return new_graph

    async def reroute(self, engine, outref, argrefs):
        """The graph is inlined if it has the static_inline flag."""
        if self.has_flags('static_inline'):
            from ..abstract import VirtualReference
            from .clone import GraphCloner
            if any(isinstance(ref, VirtualReference) for ref in argrefs):
                return None
            assert self.plain()
            new_params = [ref.node for ref in argrefs]
            cl = GraphCloner(
                inline=[(self, outref.node.graph, new_params)],
            )
            new_output = cl[self.output]
            for g in cl.remapper.graph_repl.values():
                if g:
                    engine.mng.add_graph(g)
            return engine.ref(new_output, outref.context)
        else:
            return None

    #########
    # Flags #
    #########

    def set_flags(self, *flags, **kwflags):
        """Set flags for this graph."""
        for flag in flags:
            self.flags[flag] = True
        self.flags.update(kwflags)

    def has_flags(self, *flags):
        """Check if this graph has the given flags."""
        return all(self.flags.get(flag, False) for flag in flags)

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
        return self.manager.parents[self]

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
        from ..debug.label import label
        return label(self)

    def __repr__(self) -> str:
        return repr_(self, name=str(self),
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
        self.abstract = None

    @property
    def shape(self):
        """Return the node's shape."""
        from ..abstract import AbstractArray
        a = self.abstract
        if a is not None and isinstance(a, AbstractArray):
            return a.xshape()
        else:
            return None

    @property
    def incoming(self) -> Iterable['ANFNode']:
        """Return incoming nodes in order."""
        return iter(self.inputs)

    def __str__(self) -> str:
        from ..debug.label import label
        return label(self)

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

    def match(self, node):
        """Return whether the node matches the sexp or node."""
        from .utils import sexp_to_node
        if not isinstance(node, ANFNode):
            node = sexp_to_node(node, self.graph)
        return Unification().unify(self, node)


@serializable('Apply')
class Apply(ANFNode):
    """A function application.

    This node represents the application of a function to a set of arguments.

    """

    def __init__(self, inputs: List[ANFNode], graph: 'Graph') -> None:
        """Construct an application."""
        super().__init__(inputs, APPLY, graph)

    def _serialize(self):
        return {'inputs': self.inputs,
                'graph': self.graph,
                'debug': self.debug,
                'abstract': self.abstract}

    @classmethod
    def _construct(cls):
        a = cls([], None)
        data = yield a
        a.inputs = data['inputs']
        a.graph = data['graph']
        a.debug = data['debug']
        a.abstract = data['abstract']

        if a.abstract is not None:
            def _cb():
                a.abstract = a.abstract.intern()
            return _cb

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
        app.abstract = self.abstract
        return app

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, inputs=self.inputs,
                     graph=self.graph)


@serializable('Parameter')
class Parameter(ANFNode):
    """A parameter to a function.

    Parameters are leaf nodes, since they are not the result of a function
    application, and they have no value. They are entirely defined by the graph
    they belong to.

    """

    def __init__(self, graph: Graph) -> None:
        """Construct the parameter."""
        super().__init__([], PARAMETER, graph)

    def _serialize(self):
        return {'graph': self.graph,
                'debug': self.debug,
                'abstract': self.abstract}

    @classmethod
    def _construct(cls):
        p = cls(None)
        data = yield p
        p.graph = data['graph']
        p.debug = data['debug']
        p.abstract = data['abstract']

        if p.abstract is not None:
            def _cb():
                p.abstract = p.abstract.intern()
            return _cb

    def is_parameter(self):
        """Return whether self is a Parameter."""
        return True

    def __repr__(self) -> str:
        return repr_(self, name=self.debug.debug_name, graph=self.graph)


@serializable('Constant')
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

    def _serialize(self):
        return {'value': self.value,
                'debug': self.debug,
                'abstract': self.abstract}

    @classmethod
    def _construct(cls):
        c = cls(None)
        data = yield c
        c.value = data['value']
        c.debug = data['debug']
        c.abstract = data['abstract']

        if c.abstract is not None:
            def _cb():
                c.abstract = c.abstract.intern()
            return _cb

    def is_constant(self, cls: Any = object) -> bool:
        """Return whether self is a Constant, with value of given cls."""
        return isinstance(self.value, cls)

    def is_constant_graph(self) -> bool:
        """Return whether self is a Constant with a Graph value."""
        return self.is_constant(Graph)

    def __visit__(self, fn):
        ct = Constant(noseq(fn, self.value))
        ct.abstract = self.abstract
        return ct

    def __str__(self) -> str:
        return f'_constant:{self.value}'

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


class VarNode(Special):
    """Graph node that represents a variable."""

    @property
    def __var__(self):
        return self.special


__consolidate__ = True
__all__ = [
    'ANFNode',
    'Apply',
    'Constant',
    'Graph',
    'Parameter',
    'Special',
    'VarNode',
]
