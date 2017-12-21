"""Visualization tools."""
from collections import defaultdict
from typing import Iterable

import pygraphviz as pgv

from myia.anf_ir import Graph
from myia.ir_utils import dfs


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
            plot.add_node(repr(node), label=str(node))
            if node.graph:
                subgraphs[node.graph].append(repr(node))
            for i in node.inputs:
                plot.add_node(repr(i), label=str(i))
                plot.add_edge(repr(i), repr(node))
    for graph, nodes in subgraphs.items():
        plot.add_subgraph(nodes, f"cluster_{repr(graph)}", label=str(graph))
    return plot
