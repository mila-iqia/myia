"""Implementation of a directed graph.

Used to convert a myia graph to a directed graph.
Directed graph is then visited to generate Python code or topological order of nodes.
"""
from collections import deque


class DirectedGraph:
    """Directed graph.

    Attributes:
    data: optional data associated to directed graph
    uses: mapping of vertex to output vertices (vertex -> output vertex).
    used_by: mapping of vertex to input vertices (vertex <- input vertex).
    """

    __slots__ = ("data", "uses", "used_by")

    def __init__(self, data=None):
        """Initialize.

        Arguments:
            data: optional value associated to graph (e.g. a myia graph)
        """
        self.data = data
        self.uses = {}
        self.used_by = {}

    def vertices(self):
        """Return a set of all vertices in directed graph."""
        return set(self.uses) | set(self.used_by)

    def add_arrow(self, a, b):
        """Add arrow a -> b to graph.

        Arguments:
            a: input value
            b: output value

        Returns:
            bool: True if arrow was added,
                False if arrow already exists in directed graph
        """
        # Do not add an arrow twice.
        if self.has_arrow(a, b):
            return False
        self.uses.setdefault(a, []).append(b)
        self.used_by.setdefault(b, []).append(a)
        return True

    def has(self, v):
        """Return True if given value is in graph."""
        return v in self.uses or v in self.used_by

    def has_arrow(self, a, b):
        """Return True if graph has arrow a -> b."""
        return a in self.uses and b in self.uses[a]

    def _copy(self):
        """Return a copy of the graph."""
        cp = DirectedGraph(self.data)
        cp.uses = {a: list(b) for a, b in self.uses.items()}
        cp.used_by = {b: list(a) for b, a in self.used_by.items()}
        return cp

    def pop(self, user):
        """Remove given value. Value to remove should be unused.

        Arguments:
            user: value to remove

        Returns:
            list: list of values used by removed value
                (list of `output` for all arrows `user -> output`)
        """
        assert user not in self.used_by
        related = self.uses.pop(user, ())
        for used in related:
            self.used_by[used].remove(user)
            if not self.used_by[used]:
                del self.used_by[used]
        return related

    def visit(self):
        """Visit directed graph from user to used values.

        Generate a sequence of graph values in visited order.
        """
        # Use a copy to visit graph, so that current graph is not modified.
        cp = self._copy()
        # Starts with unused nodes.
        todo = deque(n for n in cp.uses if n not in cp.used_by)
        while todo:
            element = todo.popleft()
            yield element
            todo.extend(n for n in cp.pop(element) if n not in cp.used_by)
        assert not cp.uses
        assert not cp.used_by
