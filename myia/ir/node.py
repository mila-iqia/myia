"""Graph representation."""

from myia import basics
from myia.utils import Named
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

    def replace(self, mapping, mapping_seq={}, recursive=True):
        """Replace nodes in the graph.

        This will recursively replace `node` with `mapping[node]` in
        the graph if `node` is in `mapping`.

        If `node` comes from a sequence edge, it will first look in
        `mapping_seq` for a replacement.

        If `recursive` is True, it will also replace nodes in the
        children of this graph.

        A node can be in either mapping or mapping_seq or both.
        """
        todo = [self.return_]
        seen = set()
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            for edge in list(node.edges.values()):
                if edge.label is SEQ and edge.node in mapping_seq:
                    repl = mapping_seq[edge.node]
                    if repl is None:
                        edge.node = None
                        del node.edges[SEQ]
                    else:
                        edge.node = mapping_seq[edge.node]
                elif edge.node in mapping:
                    edge.node = mapping[edge.node]
                if edge.node:
                    if edge.node.is_apply():
                        todo.append(edge.node)
                    elif (
                        recursive
                        and edge.node.is_constant_graph()
                        and edge.node.value.parent is self
                    ):
                        edge.node.value.replace(
                            mapping, mapping_seq, recursive=True
                        )

    def add_debug(self, **kwargs):
        """Add debug information.

        This is ignored if debug was not active when the graph was created.
        """
        if self.debug is not None:
            for k, v in kwargs.items():
                setattr(self.debug, k, v)


class Node:
    """Element in the compute graph for `Graph`.

    The defines the basic node with attributes common to all nodes.

    Attributes:
      abstract: Inferred type for this node, optional
      annotation: Defined type for this node, optional
    """

    __slots__ = ("abstract", "annotation", "debug", "__weakref__")

    def __init__(self):
        self.abstract = None
        self.annotation = None
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


class Edge:
    """Link between `Node` in `Graph`.

    Attributes:
      label: The label for the link, can be any object.
      node: The target node.
    """

    __slots__ = ("label", "node")

    def __init__(self, label, node):
        self.label = label
        self.node = node

    def clone(self, g, objmap):
        """Make a copy, in the context of a graph clone.

        Arguments:
          g: The graph that is cloned.
          objmap: Map of cloned objets
        """
        return Edge(self.label, self.node.clone(g, objmap))


def _edgemap(edges):
    res = {}
    for e in edges:
        assert e.label not in res
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
        self.edges = _edgemap(edges)

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
        res.edges = _edgemap(e.clone(g, objmap) for e in self.edges.values())
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
