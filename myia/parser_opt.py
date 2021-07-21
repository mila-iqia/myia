"""Parser optimizations, applicable on graph after parsing."""
from collections import deque

from myia.basics import (
    global_universe_getitem,
    global_universe_setitem,
    make_handle,
)
from myia.ir.graph_utils import toposort
from myia.ir.node import SEQ, Graph


def remove_useless_universe_getitem(g: Graph):
    """Replace universe_getitem nodes with associated values if possible.

    If a make_handle node is set in an initial universe_setitem call and used only in some
    universe_getitem nodes, then we can just replace all universe_getitem nodes with the value
    associated to make_handle, and remove initial universe_setitem and associated typeof node.
    """
    # Apply optimization to all graphs.
    seen_graphs = set()
    todo_graphs = deque([g])
    while todo_graphs:
        graph = todo_graphs.popleft()
        if graph in seen_graphs:
            continue
        seen_graphs.add(graph)
        # Iterate nodes in raw order to speed-up code
        graph_nodes = toposort(graph, reverse=True)
        # Optimize each make_handle node.
        nodes_make_handle = [
            node for node in graph_nodes if node.is_apply(make_handle)
        ]
        for n_make_handle in nodes_make_handle:
            nodes_setitem = []
            nodes_getitem = []
            for use in n_make_handle.users:
                user = use.user
                if use.label is SEQ:
                    continue
                if user.is_apply(global_universe_setitem):
                    nodes_setitem.append(user)
                else:
                    # Currently, make_handle node is used only by universe_setitem/getitem nodes.
                    assert user.is_apply(global_universe_getitem)
                    nodes_getitem.append(user)
            assert nodes_setitem
            # To optimize, make_handle must be used by:
            # - only 1 universe_setitem
            # - at least 1 universe_getitem to replace
            if nodes_getitem and len(nodes_setitem) == 1:
                (n_setitem,) = nodes_setitem
                n_value = n_setitem.inputs[1]
                # Replace universe_getitem nodes with associated value.
                for n_getitem in nodes_getitem:
                    graph.delete_seq(n_getitem)
                    n_getitem.replace(n_value)
                graph.delete_seq(n_setitem)
                graph.delete_seq(n_make_handle)
        # Optimize all other closures.
        todo_graphs.extend(
            node.value for node in graph_nodes if node.is_constant_graph()
        )


parser_opts = [remove_useless_universe_getitem]


def apply_parser_opts(g: Graph):
    """Apply all parser optimizations and return graph."""
    remove_useless_universe_getitem(g)
    return g
