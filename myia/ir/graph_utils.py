"""Utility functions for myia graph."""
from collections import deque
from typing import Any, Callable, Iterable, Set, TypeVar

from ovld import ovld

from myia.ir.node import Apply, Graph, Node
from myia.utils.directed_graph import DirectedGraph

FOLLOW = "follow"
NOFOLLOW = "nofollow"
EXCLUDE = "exclude"


T = TypeVar("T")


def always_include(node: Any) -> str:
    """Include a node in the search unconditionally."""
    return FOLLOW


def dfs(
    root: T,
    succ: Callable[[T], Iterable[T]],
    include: Callable[[T], str] = always_include,
) -> Iterable[T]:
    """Perform a depth-first search.

    Arguments:
        root: The node to start from.
        succ: A function that returns a node's successors.
        include: A function that returns whether to include a node
            in the search.
            * Return 'follow' to include the node and follow its edges.
            * Return 'nofollow' to include the node but not follow its edges.
            * Return 'exclude' to not include the node, nor follow its edges.
    """
    seen: Set[T] = set()
    to_visit = [root]
    while to_visit:
        node = to_visit.pop()
        if node in seen:
            continue
        seen.add(node)
        incl = include(node)
        if incl == FOLLOW:
            yield node
            to_visit += succ(node)
        elif incl == NOFOLLOW:
            yield node
        elif incl == EXCLUDE:
            pass
        else:
            raise ValueError(
                "include(node) must return one of: "
                '"follow", "nofollow", "exclude"'
            )


@ovld
def incoming(_: Node):
    return []


@ovld
def incoming(node: Apply):  # noqa: F811
    return [e.node for e in node.edges.values()]


@ovld
def incoming(g: Graph):  # noqa: F811
    return [g.return_, *g.parameters] if g.return_ else []


def succ_deep(node: Node) -> Iterable[Node]:
    """Follow node.incoming and graph references.

    A node's successors are its edges set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if node.is_constant_graph():
        return incoming(node.value)
    else:
        return incoming(node)


def succ_deeper(node: Node) -> Iterable[Node]:
    """Follow edges and graph references.

    Unlike `succ_deep` this visits all encountered graphs thoroughly, including
    those found through free variables.
    """
    if node.is_constant_graph():
        return incoming(node.value)
    elif getattr(node, "graph", None):
        return [*incoming(node), *incoming(node.graph)]
    else:
        return incoming(node)


def succ_incoming(node: Node) -> Iterable[Node]:
    """Follow edges."""
    return incoming(node)


def toposort(graph: Graph, reverse=False):
    """Return graph nodes in execution order.

    Arguments:
        graph: myia graph
        reverse: if True, return nodes in raw topological order,
            from user to used nodes

    Returns:
        list: graph nodes in topological order (from user to used nodes)
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
