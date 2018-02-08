"""Utilities for manipulating and inspecting the IR."""
from typing import Iterable

from myia.anf_ir import ANFNode, Constant, Graph
from myia.graph_utils import dfs as _dfs, toposort as _toposort


#######################
# Successor functions #
#######################


def succ_deep(node):
    """Follow node.incoming and graph references.

    A node's successors are its `incoming` set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if is_constant_graph(node):
        return [node.value.return_]
    else:
        return node.incoming


def succ_incoming(node):
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


##################
# Misc utilities #
##################


def is_constant_graph(x: ANFNode) -> bool:
    """Return whether x is a Constant with a Graph value."""
    return isinstance(x, Constant) and isinstance(x.value, Graph)
