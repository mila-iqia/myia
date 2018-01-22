"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.anf_ir import Graph, ANFNode


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
            to_visit.append(node.value.return_)
