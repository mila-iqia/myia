"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.anf_ir import ANFNode, Constant, Graph


def dfs(root: ANFNode, follow_graph: bool = False) -> Iterable[ANFNode]:
    """Perform a depth-first search."""
    seen = set()
    to_visit = [root]
    while to_visit:
        node = to_visit.pop()
        seen.add(node)
        yield node
        for in_ in node.incoming:
            if in_ not in seen:
                to_visit.append(in_)
        if isinstance(node.value, Graph) and follow_graph:
            if node.value.return_ not in seen:
                to_visit.append(node.value.return_)


def is_constant_graph(x: ANFNode) -> bool:
    """Return whether x is a Constant with a Graph value."""
    return isinstance(x, Constant) and isinstance(x.value, Graph)
