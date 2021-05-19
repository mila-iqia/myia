import os

from snektalk import pastevar

from myia.ir.node import FN, SEQ, Apply, Constant, Graph, Node
from myia.parser import parse
from myia.utils.info import Labeler, enable_debug


class _NodeCache:
    """Adapter for the Labeler to deal with Constant graphs.

    Copied from myia.ir.print
    """

    def __init__(self):
        self.lbl = Labeler(
            disambiguator=self._disambiguator,
            object_describer=self._constant_describer,
            reverse_order=True,
        )

    def __call__(self, node):
        if isinstance(node, Constant) and node.is_constant_graph():
            return self.lbl(node.value)
        else:
            return self.lbl(node)

    @classmethod
    def _disambiguator(cls, label, identifier):
        return f"{label}.{identifier}"

    @classmethod
    def _constant_describer(cls, node):
        if isinstance(node, Constant) and not node.is_constant_graph():
            return str(node.value)


class GraphPrinter:
    __slots__ = ("graph", "_on_node")

    __cystyle__ = open(os.path.join(os.path.dirname(__file__), "graph.css")).read()

    def __init__(self, graph: Graph, on_node=None):
        self.graph = graph
        self._on_node = on_node

    def on_node(self, data):
        if not self._on_node:
            return
        return self._on_node(data)

    @classmethod
    def __hrepr_resources__(cls, H):
        return [
            H.javascript(
                src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.17.0/cytoscape.min.js",
                export="cytoscape",
            ),
            H.javascript(
                src="https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/1.5.0/cytoscape-dagre.js",
                export="cytoscape-dagre",
            ),
            H.javascript(
                src="https://cdn.rawgit.com/cpettitt/dagre/v0.7.4/dist/dagre.js",
                export="dagre",
            ),
            H.javascript(
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

    def __hrepr__(self, H, hrepr):
        nodecache = _NodeCache()
        graphs, edges = self.get_graphs_and_edges(self.graph)
        nodes = {user for user, _, _ in edges} | {used for _, _, used in edges}
        data = []

        data += [
            {
                "data": {
                    "id": str(id(graph)),
                    "label": nodecache(graph),
                    "parent": str(id(graph.parent)) if graph.parent else None,
                },
                "classes": "function",
            }
            for graph in graphs
        ]

        data += [
            {
                "data": {
                    "id": str(id(node)),
                    "label": self.expr(node, nodecache),
                    "parent": str(id(node.graph))
                    if not node.is_constant()
                    else None,
                },
                "classes": self.get_node_class(node),
            }
            for node in nodes
            if isinstance(node, Node)
        ]

        data += [
            {
                "data": {"source": str(id(tgt)), "target": str(id(src))},
                "classes": self.get_edge_class(edge_label),
            }
            for src, edge_label, tgt in edges
        ]

        width = hrepr.config.graph_width or 800
        height = hrepr.config.graph_height or 800
        style = hrepr.config.graph_style or self.__cystyle__
        return H.div(
            style=f"width:{width}px;height:{height}px;",
            constructor="make_graph",
            options={
                "elements": data,
                "style": style,
                "layout": {"name": "dagre"},
                "on_node": self.on_node,
            },
        )

    @classmethod
    def get_graphs_and_edges(cls, g: Graph):
        all_graphs = []
        all_edges = []
        seen_graphs = set()
        seen_edges = set()
        cls.collect_graph(g, all_graphs, all_edges, seen_graphs, seen_edges)
        all_edges.reverse()
        return all_graphs, all_edges

    @classmethod
    def collect_graph(
        cls,
        graph: Graph,
        all_graphs: list,
        all_edges: list,
        seen_graphs: set,
        seen_edges: set,
    ):
        if graph not in seen_graphs:
            seen_graphs.add(graph)
            all_graphs.append(graph)
            cls.collect_edge(
                (None, None, graph.return_),
                all_graphs,
                all_edges,
                seen_graphs,
                seen_edges,
            )

    @classmethod
    def collect_edge(
        cls,
        edge: tuple,
        all_graphs: list,
        all_edges: list,
        seen_graphs: set,
        seen_edges: set,
    ):
        if edge not in seen_edges:
            seen_edges.add(edge)
            user, label, used = edge
            if user is not None:
                all_edges.append(edge)
            if isinstance(used, Apply):
                for e in used.edges.values():
                    if e.node is not None:
                        node = e.node
                        if node.is_constant_graph():
                            cls.collect_edge(
                                (used, e.label, node.value),
                                all_graphs,
                                all_edges,
                                seen_graphs,
                                seen_edges,
                            )
                            cls.collect_graph(
                                node.value,
                                all_graphs,
                                all_edges,
                                seen_graphs,
                                seen_edges,
                            )
                        else:
                            cls.collect_edge(
                                (used, e.label, node),
                                all_graphs,
                                all_edges,
                                seen_graphs,
                                seen_edges,
                            )

    @classmethod
    def expr(cls, node, nodecache):
        if isinstance(node, Apply):
            return f"{nodecache(node)} = {nodecache(node.fn)}({', '.join(nodecache(inp) for inp in node.inputs)})"
        return nodecache(node)

    @classmethod
    def get_node_class(cls, node):
        if node.is_parameter():
            return "input"
        if node.is_constant():
            return "constant"
        if node is node.graph.return_:
            return "output"
        return "intermediate"

    @classmethod
    def get_edge_class(cls, edge_label):
        if edge_label is SEQ:
            return "link-edge"
        if edge_label is FN:
            return "fn-edge"
        if isinstance(edge_label, int):
            return "input-edge"
        return None


def main():
    def f(x):
        while x:
            x = x - 1
        return x

    with enable_debug():
        graph = parse(f)
    print(GraphPrinter(graph, on_node=pastevar))
    # from hrepr import hrepr
    # hrepr.page(graph, file="output.html")


if __name__ == "__main__":
    main()
