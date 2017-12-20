"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.ir import Node


def dfs(root: Node) -> Iterable[Node]:
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
