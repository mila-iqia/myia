"""Python backend optimized.

Optimize directed graphs just before code generation.

Inputs: sequence of directed graphs to compile.

Output: a list of special dictionaries to pass to code generator to "patch" directed graphs.
Neither directed graphs nor myia graphs will be modified. Instead, code generator will use
"patch" dictionaries to generate optimized code on-the-fly.
"""
from myia.compile.backends.python.directed_graph import DirectedGraph
from myia.ir.node import Apply

TYPEOF = "typeof"
MAKE_HANDLE = "make_handle"
UNIVERSE_SETITEM = "universe_setitem"
UNIVERSE_GETITEM = "universe_getitem"


class Optimizer:
    """Optimizer class."""

    def __init__(self):
        self.skip = {}
        self.replace = {}
        self.rename = {}
        self.nonlocals = {}

    def optimize(self, directed_graphs):
        """Optimize directed graphs.

        :param directed_graphs: list of directed graphs to compile.
        :return: patch: a map of dictionaries to pass to code generator to apply optimization during code generator.
        """
        # Recursively optimize directed graphs.
        todo = list(directed_graphs)
        seen = set()
        while todo:
            dg = todo.pop(0)
            # Each directed graph should be visited once.
            assert dg not in seen
            seen.add(dg)
            self._optimize_universe_setitem(dg)
            self._optimize_universe_getitem(dg)
            todo.extend(
                element
                for element in dg.value_to_node
                if isinstance(element, DirectedGraph)
            )
        return {
            "skip": self.skip,
            "replace": self.replace,
            "rename": self.rename,
            "nonlocals": self.nonlocals,
        }

    def _optimize_universe_setitem(self, dg: DirectedGraph):
        """Optimize `universe_setitem` nodes for given directed graph."""
        # Dictionary mapping typename to myia node to assignation object
        nodes = {TYPEOF: set(), MAKE_HANDLE: set(), UNIVERSE_SETITEM: set()}
        for element in dg.value_to_node:
            if isinstance(element, Apply):
                fn = element.fn
                if fn.is_constant() and fn.value in nodes:
                    assert element not in nodes[fn.value]
                    nodes[fn.value].add(element)

        # Skip all `typeof` nodes.
        for n_typeof in nodes[TYPEOF]:
            self.skip.setdefault(dg.data, set()).add(n_typeof)

        # SKip all `make_handle` nodes.
        for n_make_handle in nodes[MAKE_HANDLE]:
            self.skip.setdefault(dg.data, set()).add(n_make_handle)

        # Replace each `universe_setitem(handle, value)` with a new node `assign(value)`.
        # New `assign` node will be labeled with handle name from `make_handle` node.
        for n_universe_setitem in nodes[UNIVERSE_SETITEM]:
            n_make_handle, n_value = n_universe_setitem.inputs
            n_assign = dg.data.apply("assign", n_value)

            assert n_universe_setitem not in self.replace
            self.replace[n_universe_setitem] = n_assign
            self.rename[n_assign] = n_make_handle
            if n_make_handle not in nodes[MAKE_HANDLE]:
                # Handle belongs ton another graph, but is re-assigned here.
                # Register it as a non-local variable.
                self.nonlocals.setdefault(dg.data, []).append(n_make_handle)

    def _optimize_universe_getitem(self, dg: DirectedGraph):
        """Optimize `universe_getitem` nodes for given directed graph."""
        nodes_universe_getitem = []
        for element in dg.value_to_node:
            if isinstance(element, Apply):
                fn = element.fn
                if fn.is_constant() and fn.value == UNIVERSE_GETITEM:
                    nodes_universe_getitem.append(element)
        # Replace each `universe_getitem(handle)` with a new node `assign(handle)`.
        # New `assign` node will be labeled with `universe_getitem` node label.
        for n_universe_getitem in nodes_universe_getitem:
            (n_make_handle,) = n_universe_getitem.inputs
            n_assign = dg.data.apply("assign", n_make_handle)
            assert n_universe_getitem not in self.replace
            self.replace[n_universe_getitem] = n_assign
            self.rename[n_assign] = n_universe_getitem
