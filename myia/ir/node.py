from myia.utils import Named

FN = Named('$fn')
SEQ = Named('$seq')


class Graph:
    def __init__(self, parent=None):
        self.parent = parent
        self.parameters = []
        self.return_ = None
        self.flags = {}
        self.varargs = False
        self.kwargs = False
        self.defaults = []
        self.kwonly = 0

    @property
    def output(self):
        if not self.return_ or 0 not in self.return_.edges:
            raise ValueError("Graph has no input")
        return self.return_.inputs[0]

    def add_parameter(self):
        p = Parameter(self)
        self.parameter.append(p)
        return p

    def constant(self, obj):
        return Constant(obj)

    def apply(self, fn, *inputs):
        edges = [
            Edge(p, i if isinstance(i, Node) else self.constant(i)) for p, i in enumerate(inputs)
        ]
        edges.append(Edge(FN, fn))

        return Apply(self edges)


class Node:
    def __init__(self, graph):
        self.graph = graph
        self.abstract = None
        self.annotation = None

    def is_apply(self, value):
        return False

    def is_parameter(self):
        return False

    def is_constant(self, cls=object):
        return False

    def is_constant_graph(self):
        return False


class Edge:
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
    def __init__(self, graph, *edges):
        super().__init__(graph)
        self.edges = edgemap(edges)

    def is_apply(self, value):
        if value is not None:
            fn = self.edges[FN]
            return fn.is_constant() and fn.value is value
        else:
            return True


class Parameter(Node):
    def is_parameter(self):
        return True


class Constant(Node):
    def __init__(self, value):
        super().__init__(None)
        self.value = value

    def is_constant(self, cls=object):
        return isinstance(self.value, cls)

    def is_constant_graph(self):
        return self.is_constant(Graph)
