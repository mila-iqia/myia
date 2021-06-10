"""Utility functions for myia graph."""
from collections import deque

from myia.ir.node import Apply, Graph
from myia.utils.directed_graph import DirectedGraph


def toposort(graph: Graph, reverse=False):
    """Return graph nodes in execution order.

    :param graph: myia graph
    :param reverse: if True, return nodes in raw topological order, from user to used nodes
    :return: graph nodes in topological order (from user to used nodes)
        or execution order (from used to user order) if reverse is True
    """
    # Use directed graph to get topological order.
    # This helps to not depend on SEQ edges.
    # It also helps to make sure a node is ordered after (or before) all nodes that use it.
    directed = DirectedGraph()
    # Use None as first user node, so that graph.return_ is always added,
    # even when it's the only one node in the graph.
    todo_arrows = deque([(None, graph.return_)])
    while todo_arrows:
        user, node = todo_arrows.popleft()
        # Add only constants and nodes that belong to this graph.
        if not node.is_constant() and node.graph is not graph:
            continue
        if not directed.add_arrow(user, node):
            continue
        if isinstance(node, Apply):
            todo_arrows.extend(
                (node, edge.node) for edge in node.edges.values()
            )
    output = list(directed.visit())
    assert output.pop(0) is None
    # If reverse is False, we return nodes in execution order.
    # Otherwise, we return default topological order (from user to used nodes).
    if not reverse:
        output.reverse()
    return output
