"""Miscellaneous utilities for debugging."""

from collections import defaultdict
from myia.debug.label import short_labeler
from myia.anf_ir_utils import \
    is_constant, is_parameter, is_apply, is_constant_graph, \
    succ_deeper, succ_incoming
from myia.graph_utils import dfs, always_include


class _Empty:
    """Bogus class, used internally by mixin."""


def mixin(target):
    """Class decorator to add methods to the target class."""
    def apply(cls):
        methods = set(dir(cls))
        methods.difference_update(set(dir(_Empty)))
        for mthd in methods:
            setattr(target, mthd, getattr(cls, mthd))
        return target
    return apply


def _same_node_shallow(n1, n2, equiv):
    # Works for Constant, Parameter and nodes previously seen
    if n1 in equiv and equiv[n1] is n2:
        return True
    elif is_constant_graph(n1) and is_constant_graph(n2):
        # Note: we provide current equiv so that nested graphs can properly
        # match their free variables, using the equiv of their parent graph.
        return isomorphic(n1.value, n2.value, equiv)
    elif is_constant(n1):
        return n1.value == n2.value
    elif is_parameter(n1):
        return False
    else:
        raise TypeError(n1)


def _same_node(n1, n2, equiv):
    # Works for Apply (when not seen previously) or other nodes
    if is_apply(n1):
        return all(_same_node_shallow(i1, i2, equiv)
                   for i1, i2 in zip(n1.inputs, n2.inputs))
    else:
        return _same_node_shallow(n1, n2, equiv)


def _same_subgraph(root1, root2, equiv):
    # Check equivalence between two subgraphs, starting from root1 and root2,
    # using the given equivalence dictionary. This is a modified version of
    # toposort that walks the two graphs in lockstep.

    done = set()
    todo = [(root1, root2)]

    while todo:
        n1, n2 = todo[-1]
        if n1 in done:
            todo.pop()
            continue
        cont = False

        s1 = list(succ_incoming(n1))
        s2 = list(succ_incoming(n2))
        if len(s1) != len(s2):
            return False
        for i, j in zip(s1, s2):
            if i not in done:
                todo.append((i, j))
                cont = True

        if cont:
            continue
        done.add(n1)

        res = _same_node(n1, n2, equiv)
        print(res, n1, n2, equiv)
        if res:
            equiv[n1] = n2
        else:
            return False

        todo.pop()

    return True


def isomorphic(g1, g2, equiv=None):
    """Return whether g1 and g2 are structurally equivalent.

    Constants are isomorphic iff they contain the same value or are isomorphic
    graphs.

    g1.return_ and g2.return_ must represent the same node under the
    isomorphism. Parameters must match in the same order.
    """
    if len(g1.parameters) != len(g2.parameters):
        return False

    prev_equiv = equiv
    equiv = dict(zip(g1.parameters, g2.parameters))
    if prev_equiv:
        equiv.update(prev_equiv)

    return _same_subgraph(g1.return_, g2.return_, equiv)


class GraphIndex:
    """Utility to map names to nodes and graphs.

    A depth first search is initiated on the given graph, and the name of each
    encountered node is mapped to the node.
    """

    def __init__(self,
                 g,
                 labeler=short_labeler,
                 succ=succ_deeper,
                 include=always_include):
        """Create a GraphIndex."""
        self.labeler = labeler
        self._index = defaultdict(set)

        self._acquire(g)

        for node in dfs(g.return_, succ, include):
            self._acquire(node)
            if node.graph:
                self._acquire(node.graph)

    def _acquire(self, obj):
        name = self.labeler.name(obj)
        if name:
            self._index[name].add(obj)

    def get_all(self, key):
        """Get all nodes/graphs corresponding to the given key."""
        return self._index[key]

    def __getitem__(self, key):
        v, = self._index[key]
        return v
