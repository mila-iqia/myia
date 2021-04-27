from types import SimpleNamespace

from myia.utils import Named
from myia.utils.info import DebugInfo

FN = Named("$fn")
SEQ = Named("$seq")


class Graph:
    def __init__(self, parent=None):
        self.parent = parent
        self.parameters = []
        self.return_ = None
        self.flags = {}
        self.varargs = False
        self.kwargs = False
        self.defaults = {}
        self.kwonly = 0
        self.debug = DebugInfo(obj=self)

    @property
    def output(self):
        if not self.return_ or 0 not in self.return_.edges:
            raise ValueError("Graph has no output")
        return self.return_.edges[0].node

    @output.setter
    def output(self, value):
        if self.return_:
            self.return_.edges[0].node = value
        else:
            self.return_ = self.apply("return_", value)
        self.return_.abstract = value.abstract
        # XXX: add typing for the "return_" primitive (or maybe not)

    def set_flags(self, **flags):
        self.flags.update(flags)

    def add_parameter(self, name):
        p = Parameter(self, name)
        self.parameters.append(p)
        return p

    def constant(self, obj):
        return Constant(obj)

    def apply(self, fn, *inputs):
        edges = [
            Edge(p, i if isinstance(i, Node) else self.constant(i))
            for p, i in enumerate(inputs)
        ]
        edges.append(
            Edge(FN, fn if isinstance(fn, Node) else self.constant(fn))
        )

        return Apply(self, *edges)

    def replace(self, mapping, mapping_seq={}):
        todo = [self.return_]
        seen = set()
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            for edge in node.edges.values():
                if edge.label is SEQ and edge.node in mapping_seq:
                    edge.node = mapping_seq[edge.node]
                elif edge.node in mapping:
                    edge.node = mapping[edge.node]
                if edge.node and edge.node.is_apply():
                    todo.append(edge.node)

    def add_debug(self, **kwargs):
        if self.debug is not None:
            for k, v in kwargs.items():
                setattr(self.debug, k, v)


class Node:
    __slots__ = ("abstract", "annotation", "debug", "__weakref__")

    def __init__(self):
        self.abstract = None
        self.annotation = None
        self.debug = DebugInfo(obj=self)

    def is_apply(self, value=None):
        return False

    def is_parameter(self):
        return False

    def is_constant(self, cls=object):
        return False

    def is_constant_graph(self):
        return False

    def add_annotation(self, annotation):
        self.annotation = annotation

    def add_debug(self, **kwargs):
        if self.debug is not None:
            for k, v in kwargs.items():
                setattr(self.debug, k, v)


class Edge:
    __slots__ = ("label", "node")

    def __init__(self, label, node):
        self.label = label
        self.node = node


def edgemap(edges):
    res = {}
    for e in edges:
        assert e.label not in res
        res[e.label] = e

    return res


class Apply(Node):
    __slots__ = ("edges", "graph")

    def __init__(self, graph, *edges):
        super().__init__()
        self.graph = graph
        self.edges = edgemap(edges)

    def is_apply(self, value=None):
        if value is not None:
            fn = self.edges[FN]
            return fn.is_constant() and fn.value is value
        else:
            return True

    def add_edge(self, label, node):
        assert label not in self.edges
        e = Edge(label, node)
        self.edges[label] = e

    @property
    def fn(self):
        return self.edges[FN].node

    @property
    def inputs(self):
        i = 0
        res = []
        while i in self.edges:
            res.append(self.edges[i].node)
            i += 1
        return tuple(res)


class Parameter(Node):
    __slots__ = ("graph", "name")

    def __init__(self, graph, name):
        super().__init__()
        self.name = name
        self.add_debug(name=name)
        self.graph = graph

    def is_parameter(self):
        return True


class Constant(Node):
    __slots__ = ("value",)

    def __init__(self, value):
        super().__init__()
        self.value = value

    def is_constant(self, cls=object):
        return isinstance(self.value, cls)

    def is_constant_graph(self):
        return self.is_constant(Graph)
