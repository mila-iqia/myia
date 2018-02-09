"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable, Callable, Set

from myia.anf_ir import ANFNode, Constant, Graph
from myia.graph_utils import dfs as _dfs, toposort as _toposort


#######################
# Successor functions #
#######################


def succ_deep(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    A node's successors are its `incoming` set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if is_constant_graph(node):
        return [node.value.return_]
    else:
        return node.incoming


def succ_incoming(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming."""
    return node.incoming


def succ_stop_at_fv(graph):
    """Follow node.incoming for nodes that belong to graph.

    This successor function does not follow nodes that belong to other graphs
    (i.e. free variables). These nodes can be successors, however.
    """
    def succ(node):
        if node.graph is graph:
            return node.incoming
        else:
            return []


def succ_bidirectional(scope: Set[Graph]) -> Callable:
    """Follow node.incoming, node.users and graph references.

    `succ_bidirectional` will only return nodes that belong to the given
    set of graphs.
    """
    def succ(node: ANFNode) -> Iterable[ANFNode]:
        rval = set(node.incoming) | {u for u, _ in node.uses}
        if is_constant_graph(node):
            rval.add(node.value.return_)
        return {x for x in rval if x.graph in scope}

    return succ


#####################
# Search algorithms #
#####################


def dfs(root: ANFNode, follow_graph: bool = False) -> Iterable[ANFNode]:
    """Perform a depth-first search."""
    return _dfs(root, succ_deep if follow_graph else succ_incoming)


def toposort(root: ANFNode) -> Iterable[ANFNode]:
    """Order the nodes topologically."""
    return _toposort(root, succ_incoming)


def accessible_graphs(root: Graph) -> Set[Graph]:
    """Return all Graphs accessible from root."""
    return {root} | {x.value for x in dfs(root.return_, True)
                     if is_constant_graph(x)}


###########
# Cleanup #
###########


def destroy_disconnected_nodes(root: Graph) -> None:
    """Remove dead nodes that belong to the graphs accessible from root.

    The `uses` set of a node may keep alive some nodes that are not connected
    to the output of a graph (e.g. `_, x = pair`, where `_` is unused). These
    nodes are removed by this function.
    """
    # We restrict ourselves to graphs accessible from root, otherwise we may
    # accidentally destroy nodes from other graphs that are users of the
    # constants we use.
    cov = accessible_graphs(root)
    live = dfs(root.return_, True)
    total = _dfs(root.return_, succ_bidirectional(cov))
    dead = set(total) - set(live)
    for node in dead:
        node.inputs.clear()  # type: ignore


##################
# Misc utilities #
##################


def is_constant_graph(x: ANFNode) -> bool:
    """Return whether x is a Constant with a Graph value."""
    return isinstance(x, Constant) and isinstance(x.value, Graph)
