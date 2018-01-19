"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.ir import Node
from myia.anf_ir import Graph


def dfs(root: Node, follow_graph: bool = False) -> Iterable[Node]:
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


def find_graphs(root: Node) -> Iterable[Graph]:
    """Given a node, find all the graphs involved in its definition."""
    seen = set()
    for node in dfs(root, follow_graph=True):
        if isinstance(node.graph, Graph) and node.graph not in seen:
            seen.add(node.graph)
            yield node.graph
        elif isinstance(node.value, Graph) and node.value not in seen:
            seen.add(node.value)
            yield node.value
