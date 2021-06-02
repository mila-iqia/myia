"""Parser optimizations, applicable on graph after parsing."""
from collections import deque

from myia.basics import (
    global_universe_getitem,
    global_universe_setitem,
    make_handle,
)
from myia.ir.graph_utils import get_node_users, toposort
from myia.ir.node import SEQ, Apply, Graph


def _replace_apply_node(root_graph: Graph, node: Apply, new_node):
    """Replace node by new_node in given graph.

    :param root_graph: graph in which to make replacements.
        Replacement will be recursively applied to graph and all children closures.
    :param node: node to replace
    :param new_node: node to replace with. May be None to delete node.
    """
    mapping = {node: new_node}
    mapping_seq = {node: (node.edges[SEQ].node if SEQ in node.edges else None)}
    root_graph.replace(mapping, mapping_seq, recursive=True)


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
            for user in get_node_users(n_make_handle):
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
                    _replace_apply_node(graph, n_getitem, n_value)
                # Just remove universe_setitem.
                # typeof and make_handle should be deleted with it.
                mapping_setitem = {}
                mapping_setitem_seq = {
                    n_setitem: (
                        n_setitem.edges[SEQ].node
                        if SEQ in n_setitem.edges
                        else None
                    )
                }
                graph.replace(
                    mapping_setitem, mapping_setitem_seq, recursive=True
                )
        # Optimize all other closures.
        todo_graphs.extend(
            node.value for node in graph_nodes if node.is_constant_graph()
        )


parser_opts = [remove_useless_universe_getitem]
