"""Visualization tools."""
from collections import defaultdict
from typing import Iterable

import pygraphviz as pgv

from myia.anf_ir import Graph, Constant, Apply, Parameter
from myia.primops import Primitive
from myia.ir_utils import dfs


def name(node):
    if isinstance(node, Constant):
        if isinstance(node.value, Primitive):
            return node.value.__class__.__name__.lower()
        elif isinstance(node.value, Graph):
            return node.value.debug.debug_name
        return str(node)
    elif isinstance(node, Graph):
        return str(node)
    elif isinstance(node, Apply):
        return 'apply'
    elif isinstance(node, Parameter):
        return str(node)
    else:
        raise ValueError


def draw(graphs: Iterable[Graph]) -> pgv.AGraph:
    """Plot a graph using Graphviz.

    This function will walk the graph starting from the return nodes of each
    graph, plotting all the nodes it finds. It will draw boxes around each
    graph. The string representations of nodes and graphs will be used as
    labels.

    """
    plot = pgv.AGraph(strict=False, directed=True)
    subgraphs = defaultdict(list)
    for graph in graphs:
        for node in dfs(graph.return_):
            plot.add_node(repr(node),
                          label=f'{node.debug.debug_name}: {name(node)}')
            if node.graph:
                subgraphs[node.graph].append(repr(node))
            for i in node.inputs:
                plot.add_node(repr(i),
                              label=f'{node.debug.debug_name}: {name(i)}')
                plot.add_edge(repr(i), repr(node))
    for graph, nodes in subgraphs.items():
        plot.add_subgraph(nodes, f"cluster_{repr(graph)}", label=name(graph))
    return plot
