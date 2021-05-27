"""Utility functions for myia graph."""
from myia.ir.node import Apply, Graph
from myia.utils.directed_graph import DirectedGraph


def toposort(graph: Graph, reverse=True):
    """Return graph nodes in topological order.

    :param graph: myia graph
    :param reverse: if True, return nodes in execution order (default)
    :return: graphes nodes in topological order (from user to used nodes)
        or execution order (from used to user order) if reverse is True
    """
    # Use directed graph to get topological order.
    # This helps to not depend on SEQ edges.
    # It also helps to make sure a node is ordered after (or before) all nodes that use it.
    directed = DirectedGraph()
    # Use None as first user node, so that graph.return_ is always added,
    # even when it's the only one node in the graph.
    todo_arrows = [(None, graph.return_)]
    while todo_arrows:
        user, node = todo_arrows.pop(0)
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
    if reverse:
        output.reverse()
    return output


def get_node_users(node, graph=None, recursive=True):
    """Get all apply nodes that use given node as function or inputs.

    :param node: node to search users
    :param graph: graph where to search users.
        If None, use node graph (node must not be a constant)
    :param recursive: if True, recursively look for users in graph closures
    :return: list of node users
    """
    if graph is None:
        assert (
            not node.is_constant()
        ), "A graph must be specified to find constant node users."
        graph = node.graph
    users = []
    seen_graphs = set()
    todo_graphs = [graph]
    while todo_graphs:
        g = todo_graphs.pop(0)
        if g not in seen_graphs:
            seen_graphs.add(g)
            for candidate in toposort(g, reverse=False):
                if isinstance(candidate, Apply) and (
                    node is candidate.fn or node in candidate.inputs
                ):
                    users.append(candidate)
                elif recursive and candidate.is_constant_graph():
                    todo_graphs.append(candidate.value)
    return users
