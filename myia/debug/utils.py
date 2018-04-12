"""Miscellaneous utilities for debugging."""

from collections import defaultdict
from myia.debug.label import short_labeler
from myia.anf_ir import Graph
from myia.anf_ir_utils import \
    is_constant, is_parameter, is_apply, succ_deeper
from myia.graph_utils import dfs, always_include


class _Empty:
    """Bogus class, used internally by mixin."""

    pass


def mixin(target):
    """Class decorator to add methods to the target class."""
    def apply(cls):
        methods = set(dir(cls))
        methods.difference_update(set(dir(_Empty)))
        for mthd in methods:
            setattr(target, mthd, getattr(cls, mthd))
        return target
    return apply


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

    def same(n1, n2):
        if n1 in equiv:
            return equiv[n1] is n2
        if type(n1) is not type(n2):
            return False
        if is_constant(n1):
            return same(n1.value, n2.value)
            # return n1.value == n2.value
        elif is_parameter(n1):
            return False
        elif is_apply(n1):
            success = all(same(i1, i2) for i1, i2 in zip(n1.inputs, n2.inputs))
            if success:
                equiv[n1] = n2
            return success
        elif isinstance(n1, Graph):
            return isomorphic(n1, n2, equiv)
        else:
            return n1 == n2

    return same(g1.return_, g2.return_)


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
