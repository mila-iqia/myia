"""Implementation of a directed graph.

Used to convert a myia graph to a directed graph.
Directed graph is then visited to generate Python code.
"""


class DirectedNode:
    """Helper class to represent a node in a directed graph."""

    def __init__(self, value):
        """Initialize.

        :param value: value associated to the node (e.g. an apply or a myia graph)
        """
        self.value = value


class DirectedGraph:
    """Directed graph."""

    def __init__(self, data=None):
        """Initialize.

        :param data: value associated to graph (currently, a myia graph)
        """
        self.data = data
        self.uses = {}
        self.used_by = {}
        self.value_to_node = {}

    def add_arrow(self, a, b):
        """Add arrow a -> b to graph. Values a and b are added if they don't exist.

        :param a: input value
        :param b: output value
        """
        if a not in self.value_to_node:
            self.value_to_node[a] = DirectedNode(a)

        if b not in self.value_to_node:
            self.value_to_node[b] = DirectedNode(b)

        node_a = self.value_to_node[a]
        node_b = self.value_to_node[b]
        self.uses.setdefault(node_a, []).append(node_b)
        self.used_by.setdefault(node_b, []).append(node_a)

    def has(self, v):
        """Return True if given value is in graph."""
        return v in self.value_to_node

    def is_unused(self, v):
        """Return True if there is no arrow pointing to given value."""
        return (
            v not in self.value_to_node
            or self.value_to_node[v] not in self.used_by
        )

    def replace(self, from_value, to_value):
        """Replace a value with another one in the graph.

        Currently used to recursively replace a closure with corresponding directed graph.

        :param from_value: value to replace
        :param to_value: new value
        """
        assert from_value in self.value_to_node
        node = self.value_to_node.pop(from_value)
        node.value = to_value
        self.value_to_node[to_value] = node

    def _copy(self):
        """Return a copy of the graph."""
        cp = DirectedGraph(self.data)
        cp.uses = {a: list(b) for a, b in self.uses.items()}
        cp.used_by = {b: list(a) for b, a in self.used_by.items()}
        cp.value_to_node = self.value_to_node.copy()
        return cp

    def pop(self, user):
        """Remove given value. Value to remove should be unused.

        :param user: value to remove
        :return: list of values used by removed value
            (list of `output` for all arrows `user -> output`)
        """
        if self.value_to_node[user] in self.uses:
            user_node = self.value_to_node[user]
            assert user_node not in self.used_by
            related_nodes = self.uses.pop(user_node)
            for used_node in related_nodes:
                self.used_by[used_node].remove(user_node)
                if not self.used_by[used_node]:
                    del self.used_by[used_node]
            del self.value_to_node[user]
            return [node.value for node in related_nodes]
        return []

    def visit(self):
        """Visit directed graph from user to used values.

        Use `None` value to get the first unused values.

        :return: generator: sequence of graph values in visited order.
        """
        assert self.has(None)
        cp = self._copy()
        todo = cp.pop(None)
        while todo:
            element = todo.pop(0)
            yield element
            related = cp.pop(element)
            todo.extend(used for used in related if cp.is_unused(used))
        assert not cp.uses
        assert not cp.used_by
