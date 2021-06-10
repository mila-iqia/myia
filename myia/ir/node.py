"""Graph representation."""

import weakref

from myia import basics
from myia.utils import Named, myia_hrepr_resources
from myia.utils.info import clone_debug, make_debug

FN = Named("$fn")
SEQ = Named("$seq")


class Graph:
    """Represents block of computation with arguments.

    Can optionally represent keyword arguments and default argument values.
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.parameters = []
        self.return_ = None
        self.flags = {}
        self.varargs = False
        self.kwargs = False
        self.defaults = {}
        self.posonly = 0
        self.kwonly = 0
        self.debug = make_debug(obj=self)

    @property
    def output(self):
        """The output expression.

        If modified, this will replace the entire graph.
        """
        if not self.return_ or 0 not in self.return_.edges:
            raise ValueError("Graph has no output")
        return self.return_.edges[0].node

    @output.setter
    def output(self, value):
        if self.return_:
            self.return_.edges[0].node = value
        else:
            self.return_ = self.apply(basics.return_, value)
        self.return_.abstract = value.abstract
        # XXX: add typing for the "return_" primitive (or maybe not)

    def set_flags(self, **flags):
        """Update the flags."""
        self.flags.update(flags)

    def add_parameter(self, name):
        """Append a parameter."""
        p = Parameter(self, name)
        self.parameters.append(p)
        return p

    def constant(self, obj):
        """Create a Constant."""
        return Constant(obj)

    def apply(self, fn, *inputs):
        """Create an Apply node.

        Arguments:
          fn: The function to call.
          inputs: The function inputs, if any.
        """
        edges = [
            Edge(p, i if isinstance(i, Node) else self.constant(i))
            for p, i in enumerate(inputs)
        ]
        edges.append(
            Edge(FN, fn if isinstance(fn, Node) else self.constant(fn))
        )

        return Apply(self, *edges)

    def clone(self, objmap=None):
        """Make a copy of this graph."""
        res = Graph(self.parent)
        if objmap is None:
            objmap = {self: res}
        elif self in objmap:
            return objmap[self]
        else:
            objmap[self] = res
        res.parameters = [p.clone(self, objmap) for p in self.parameters]
        res.flags = self.flags.copy()
        if self.return_:
            res.return_ = self.return_.clone(self, objmap)
        res.varargs = (
            self.varargs.clone(self, objmap) if self.varargs else self.varargs
        )
        res.kwargs = (
            self.kwargs.clone(self, objmap) if self.kwargs else self.kwargs
        )
        res.defaults = self.defaults
        res.kwonly = self.kwonly
        res.debug = clone_debug(self.debug, objmap)
        return res

    def replace_node(self, node, lbl, repl, *, recursive=True):
        """Replace a node by another in this graph.

        This will replace every use of `node` with label `lbl` in this
        graph (and subgraphs if `recursive` is `True`) by `repl`.

        If `lbl` is None then this will replace all uses.
        """
        for use in list(node.users):
            if recursive or use.user.graph is self:
                if lbl is None or use.label == lbl:
                    use.node = repl

    def delete_seq(self, node):
        """Remove a node from all sequence chains."""
        fwd = node.edges.get(SEQ, None)
        if fwd is not None:
            fwd = fwd.node
        for use in list(node.users):
            if use.label is SEQ:
                if fwd is None:
                    del use.user.edges[SEQ]
                else:
                    use.node = fwd

    def add_debug(self, **kwargs):
        """Add debug information.

        This is ignored if debug was not active when the graph was created.
        """
        if self.debug is not None:
            for k, v in kwargs.items():
                setattr(self.debug, k, v)

    __hrepr_resources__ = myia_hrepr_resources

    def __hrepr_short__(self, H, hrepr):
        from .print import global_labeler

        return H.atom["myia-Graph"](global_labeler(self))

    def __hrepr__(self, H, hrepr):
        if (
            hrepr.state.depth == 0 or hrepr.config.graph_expand
        ) and not hrepr.config.short:
            from .visualization.graph_printer import GraphPrinter

            return hrepr(
                GraphPrinter(
                    self,
                    show_fn_constants=False,
                    show_args=False,
                    link_fn_graphs=True,
                    link_inp_graphs=True,
                )
            )
        else:
            # Falls back to hrepr_short
            return NotImplemented

    def __str__(self):
        from .print import global_labeler

        name = type(self).__name__
        return f"{name}({global_labeler(self)})"

    __repr__ = __str__


class Node:
    """Element in the compute graph for `Graph`.

    The defines the basic node with attributes common to all nodes.

    Attributes:
      abstract: Inferred type for this node, optional
      annotation: Defined type for this node, optional
    """

    __slots__ = ("abstract", "annotation", "users", "debug", "__weakref__")

    def __init__(self):
        self.abstract = None
        self.annotation = None
        self.users = weakref.WeakSet()
        self.debug = make_debug(obj=self)

    def is_apply(self, value=None):
        """Check if this node is an `Apply` node.

        If `value` is not None, it will only return True if it is an
        apply of the specified function, otherwise returns True if
        it's an `Apply` node.
        """
        return False

    def is_parameter(self):
        """Check if this node is a `Parameter`."""
        return False

    def is_constant(self, cls=object):
        """Check if this node is a `Constant`."""
        return False

    def is_constant_graph(self):
        """Check if this node is a graph."""
        return False

    def add_debug(self, **kwargs):
        """Add debug information.

        This is ignored if debug was not active when the node was created.
        """
        if self.debug is not None:
            for k, v in kwargs.items():
                setattr(self.debug, k, v)

    def _copy_fields(self, old, objmap):
        self.abstract = old.abstract
        self.annotation = old.annotation
        self.debug = clone_debug(old.debug, objmap)

    def __hrepr_short__(self, H, hrepr):
        from .print import global_labeler

        typ = type(self).__name__
        return H.atom["myia-Node", f"myia-{typ}"](
            global_labeler.informative(self, hide_anonymous=False)
        )

    def __str__(self):
        from .print import global_labeler

        name = type(self).__name__
        lbl = global_labeler.informative(self, hide_anonymous=False)
        return f"{name}({lbl})"

    __repr__ = __str__


class Edge:
    """Link between `Node` in `Graph`.

    Attributes:
      label: The label for the link, can be any object.
      node: The target node.
      user: The node that uses this Edge.
    """

    __slots__ = ("label", "_node", "_user", "__weakref__")

    def __init__(self, label, node):
        self.label = label
        self._node = node
        self._node.users.add(self)
        self._user = None

    def clone(self, g, objmap):
        """Make a copy, in the context of a graph clone.

        Arguments:
          g: The graph that is cloned.
          objmap: Map of cloned objets
        """
        return Edge(self.label, self.node.clone(g, objmap))

    @property
    def node(self):
        """The target node."""
        return self._node

    @node.setter
    def node(self, node):
        self._node.users.remove(self)
        self._node = node
        self._node.users.add(self)

    @property
    def user(self):
        """Returns the node that has this edge in its edges."""
        if self._user:
            return self._user()

    @user.setter
    def user(self, node):
        self._user = weakref.ref(node)


def _edgemap(edges, self):
    res = {}
    for e in edges:
        assert e.label not in res
        e.user = self
        res[e.label] = e

    return res


class Apply(Node):
    """Nodes that represent the application of a computation.

    Attributes:
      edges: The links for function and arguments.
      graph: The graph that this node belongs to.

    Note: see also `Node` for common properties.
    """

    __slots__ = ("edges", "graph")

    def __init__(self, graph, *edges):
        super().__init__()
        self.graph = graph
        self.edges = _edgemap(edges, self)

    def is_apply(self, value=None):
        """See `Node.is_apply`."""
        if value is not None:
            fn = self.edges[FN].node
            return fn.is_constant() and fn.value is value
        else:
            return True

    def add_edge(self, label, node):
        """Add an incoming edge to this node."""
        assert label not in self.edges
        e = Edge(label, node)
        e.user = self
        self.edges[label] = e

    def append_input(self, node):
        """Add a node at the end of the input list.

        This assumes that the inputs are labelled numerically starting
        from 0 and have no gaps.
        """
        i = 0
        while i in self.edges:
            i += 1
        self.add_edge(i, node)

    @property
    def fn(self):
        """The function that this Apply calls."""
        return self.edges[FN].node

    @property
    def seq(self):
        """The previous element in the sequence chain (or None)."""
        val = self.edges.get(SEQ, None)
        return val if val is None else val.node

    @property
    def inputs(self):
        """The tuple of inputs for this apply.

        If you want to modify inputs you need to interact with edges
        since this is a view only.
        """
        i = 0
        res = []
        while i in self.edges:
            res.append(self.edges[i].node)
            i += 1
        return tuple(res)

    def clone(self, g, objmap):
        """Copy a node in the context of a graph clone.

        Arguments:
          g: The graph that is cloned.
          objmap: Cloned object map.
        """
        if self in objmap:
            return objmap[self]
        if self.graph is not g:
            return self
        res = Apply(objmap[g])
        objmap[self] = res
        res.edges = _edgemap(
            (e.clone(g, objmap) for e in self.edges.values()), res
        )
        res._copy_fields(self, objmap)
        return res


class Parameter(Node):
    """Node that represents a parameter for a `Graph`.

    Attributes:
      graph: the graph that this parameter is for.
      name: the name of this parameter, optional.

    Note: see also `Node` for common properties.
    """

    __slots__ = ("graph", "name")

    def __init__(self, graph, name):
        super().__init__()
        self.name = name
        self.add_debug(name=name)
        self.graph = graph

    def is_parameter(self):
        """See `Node.is_parameter`."""
        return True

    def clone(self, g, objmap):
        """Copy a node in the context of a graph clone.

        Arguments:
          g: The graph that is cloned.
          objmap: Cloned object map.
        """
        if self in objmap:
            return objmap[self]
        if self.graph is not g:
            return self
        res = Parameter(g, self.name)
        objmap[self] = res
        res._copy_fields(self, objmap)
        return res


class Constant(Node):
    """Node that represents constant values.

    Attributes:
      value: The constant value.

    Note: see also `Node` for common properties.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        super().__init__()
        self.value = value

    def is_constant(self, cls=object):
        """See `Node.is_constant`."""
        return isinstance(self.value, cls)

    def is_constant_graph(self):
        """See `Node.is_constant_graph`."""
        return self.is_constant(Graph)

    def clone(self, g, objmap):
        """Copy a node in the context of a graph clone.

        Arguments:
          g: The graph that is cloned.
          objmap: Cloned object map.
        """
        if self in objmap:
            return objmap[self]
        if self.is_constant_graph() and self.value.parent is g:
            res = Constant(self.value.clone(objmap))
        else:
            res = Constant(self.value)
        objmap[self] = res
        res._copy_fields(self, objmap)
        return res
