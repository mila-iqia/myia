import os
from collections import defaultdict
from snektalk import pastevar
from hrepr import hrepr, Hrepr
from myia.ir.node import Graph
import io
from myia.compile.backends.python.python import compile_graph
from myia.ir.print import str_graph, _NodeCache
from myia.parser import parse
from myia.utils.info import enable_debug
from myia.ir.node import SEQ, FN


class GraphHrepr:
    def __init__(self, graph: Graph, on_node=None):
        self.graph_to_edges = self.get_simplified(graph)
        self.nodecache = _NodeCache()
        self._on_node = on_node

    def repr(self, node):
        assert node.is_apply()
        return f"{self.nodecache(node)} = {self.nodecache(node.fn)}({', '.join(self.nodecache(inp) for inp in node.inputs)})"

    @classmethod
    def get_simplified(cls, g: Graph):
        graph_to_edges = {}
        seen_edges = set()
        todo_graphs = [g]
        while todo_graphs:
            graph = todo_graphs.pop(0)
            if graph in graph_to_edges:
                continue
            edges = []
            graph_to_edges[graph] = edges
            todo_edges = [(None, None, graph.return_)]
            while todo_edges:
                edge = todo_edges.pop(0)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                user, edge_label, used = edge
                assert used.is_apply()
                if user is not None:
                    edges.append(edge)
                for e in used.edges.values():
                    if e.node is not None:
                        node = e.node
                        if node.is_apply():
                            todo_edges.append((used, e.label, node))
                        elif node.is_constant_graph():
                            todo_graphs.append(node.value)
        return graph_to_edges

    @classmethod
    def get_node_class(cls, node):
        if node is node.graph.return_:
            return "output"
        return type(node).__name__.lower()

    @classmethod
    def get_edge_class(cls, edge_label):
        if edge_label is SEQ:
            return "link-edge"
        if edge_label is FN:
            return "fn-edge"
        if isinstance(edge_label, int):
            return "input-edge"
        return None

    def on_node(self, data):
        if not self._on_node:
            return
        return self._on_node(data)


class GraphMixin(Hrepr):

    @classmethod
    def cystyle(cls):
        return open(os.path.join(os.path.dirname(__file__), "graph.css")).read()

    def hrepr_resources(self, graph_cls: Graph):
        h = self.H
        return [
            h.javascript(
                src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.17.0/cytoscape.min.js",
                export="cytoscape",
            ),
            h.javascript(
                src="https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/1.5.0/cytoscape-dagre.js",
                export="cytoscape-dagre",
            ),
            h.javascript(
                src="https://cdn.rawgit.com/cpettitt/dagre/v0.7.4/dist/dagre.js",
                export="dagre",
            ),
            h.javascript(
                """
                // Load dagre layout
                cydagre(cytoscape, dagre);

                function make_graph(element, options) {
                    options.container = element;
                    this.cy = cytoscape(options);
                    if (options.on_node) {
                        this.cy.on('click', 'node', function(evt){
                            options.on_node(evt.target.data());
                        });
                    }
                }
                """,
                require={
                    "cytoscape": "cytoscape",
                    "dagre": "dagre",
                    "cytoscape-dagre": "cydagre",
                },
                export="make_graph",
            ),
        ]

    def hrepr(self, graph: Graph):
        h = self.H
        wrapper = GraphHrepr(graph, on_node=pastevar)
        width = self.config.graph_width or 800
        height = self.config.graph_height or 800
        style = self.config.graph_style or self.cystyle()
        data = []

        for graph, edges in wrapper.graph_to_edges.items():
            nodes = {user for user, _, _ in edges} | {used for _, _, used in edges}
            data += [{
                "data": {
                    "id": str(id(graph)),
                    "label": wrapper.nodecache(graph),
                    "parent": str(id(graph.parent)) if graph.parent else None
                },
                "classes": "function",
            }]
            data += [{
                "data": {"id": str(id(node)), "label": wrapper.repr(node), "parent": str(id(graph))},
                "classes": wrapper.get_node_class(node),
            } for node in nodes]
            data += [{
                "data": {"source": str(id(tgt)), "target": str(id(src))},
                "classes": wrapper.get_edge_class(edge_label),
            } for src, edge_label, tgt in edges]

        return h.div(
            style=f"width:{width}px;height:{height}px;",
            constructor="make_graph",
            options={
                "elements": data,
                "style": style,
                "layout": {"name": "dagre"},
                "on_node": wrapper.on_node,
            },
        )


def parse_graph(function, debug=True):
    if debug:
        with enable_debug():
            graph = parse(function)
    else:
        graph = parse(function)
    return graph


def main():
    hrepr.configure(mixins=GraphMixin)

    def f(x):
        while x:
            x = x - 1
        return x

    graph = parse_graph(f)
    print(graph)


if __name__ == "__main__":
    main()
