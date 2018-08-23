"""Miscellaneous utilities for debugging."""

import types
from collections import defaultdict

from ..graph_utils import always_include, dfs
from ..ir.utils import succ_deeper

from .label import short_labeler


class _Empty:
    """Bogus class, used internally by mixin."""


def mixin(target):
    """Class decorator to add methods to the target class."""
    def apply(cls):
        methods = set(dir(cls))
        methods.difference_update(set(dir(_Empty)))
        for method_name in methods:
            mthd = getattr(cls, method_name)
            if isinstance(mthd, types.MethodType):
                mthd = classmethod(mthd.__func__)
            setattr(target, method_name, mthd)
        return target
    return apply


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
