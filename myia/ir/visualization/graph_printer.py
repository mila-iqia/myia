import os

from myia.ir import Apply, Constant, Graph, Node
from myia.ir.node import FN, SEQ
from myia.utils.info import Labeler


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
            return repr(node.value)


class _GraphCollector:
    def __init__(self, g: Graph, show_constants=True):
        self.all_graphs = []
        self.all_edges = []
        self._show_constants = show_constants
        self._seen_graphs = set()
        self._seen_edges = set()
        self._collect_graph(g)
        self.all_edges.reverse()

    def _collect_graph(self, graph: Graph):
        if graph not in self._seen_graphs:
            self._seen_graphs.add(graph)
            self.all_graphs.append(graph)
            self._collect_edge((None, None, graph.return_))

    def _collect_edge(self, edge: tuple):
        if edge not in self._seen_edges:
            self._seen_edges.add(edge)
            user, label, used = edge
            if user is not None:
                self.all_edges.append(edge)
            if isinstance(used, Apply):
                for e in used.edges.values():
                    if e.node is not None:
                        node = e.node
                        if node.is_constant_graph():
                            self._collect_edge((used, e.label, node.value))
                            self._collect_graph(node.value)
                        elif self._show_constants or not node.is_constant():
                            self._collect_edge((used, e.label, node))


class GraphPrinter:
    __slots__ = ("graphs", "edges", "_on_node", "_show_constants")

    __cystyle__ = open(os.path.join(os.path.dirname(__file__), "graph.css")).read()

    def __init__(self, graph: Graph, *, on_node=None, show_constants=True):
        collector = _GraphCollector(graph, show_constants=show_constants)
        self.graphs = collector.all_graphs
        self.edges = collector.all_edges
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
        nodes = {user for user, _, _ in self.edges} | {used for _, _, used in self.edges}
        data = []

        # Graphs.
        data += [
            {
                "data": {
                    "id": str(id(graph)),
                    "label": nodecache(graph),
                    "parent": str(id(graph.parent)) if graph.parent else None,
                },
                "classes": "function",
            }
            for graph in self.graphs
        ]

        # Nodes.
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

        # Edges.
        data += [
            {
                "data": {"source": str(id(tgt)), "target": str(id(src)), "label": str(edge_label)},
                "classes": self.get_edge_class(edge_label),
            }
            for src, edge_label, tgt in self.edges
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