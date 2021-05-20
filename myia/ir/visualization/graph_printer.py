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


class GraphPrinter:
    __slots__ = ("graphs", "nodes", "edges", "_on_node", "_lbl")

    __cystyle__ = open(os.path.join(os.path.dirname(__file__), "graph.css")).read()

    def __init__(self, graph: Graph, *, on_node=None, show_constants=True, link_fn_graphs=True, link_inp_graphs=True):
        self.graphs, self.nodes, self.edges = self.collect_myia_elements(graph, show_constants, link_fn_graphs, link_inp_graphs)
        self._on_node = on_node
        self._lbl = _NodeCache()

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
        data = []

        # Graphs.
        data += [
            {
                "data": {
                    "id": str(id(graph)),
                    "label": self._lbl(graph),
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
                    "label": self.expr(node),
                    "parent": str(id(node.graph))
                    if not node.is_constant()
                    else None,
                },
                "classes": self.get_node_class(node),
            }
            for node in self.nodes
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
    def collect_myia_elements(cls, g: Graph, show_constants=True, link_fn_graphs=True, link_inp_graphs=True):
        all_graphs = []
        all_nodes = []
        all_edges = []
        seen_graphs = set()
        seen_nodes = set()
        seen_edges = set()
        todo_graphs = [g]
        while todo_graphs:
            graph = todo_graphs.pop(0)
            if graph in seen_graphs:
                continue
            seen_graphs.add(graph)
            all_graphs.append(graph)
            for p in graph.parameters:
                if p not in seen_nodes:
                    seen_nodes.add(p)
                    all_nodes.append(p)
            todo_edges = [(None, None, graph.return_)]
            while todo_edges:
                edge = todo_edges.pop(0)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                user, label, used = edge
                if user is not None:
                    all_edges.append(edge)
                if isinstance(used, Node) and used not in seen_nodes:
                    seen_nodes.add(used)
                    all_nodes.append(used)
                if isinstance(used, Apply):
                    inputs = set(used.inputs)
                    for e in used.edges.values():
                        node = e.node
                        if node.is_constant_graph():
                            todo_graphs.append(node.value)
                            if ((link_fn_graphs and node is used.fn) or (link_inp_graphs and node in inputs)):
                                todo_edges.append((used, e.label, node.value))
                        elif show_constants or not node.is_constant():
                            todo_edges.append((used, e.label, node))
        return all_graphs, all_nodes, all_edges

    def expr(self, node):
        if isinstance(node, Apply):
            return f"{self._lbl(node)} = {self._lbl(node.fn)}({', '.join(self._lbl(inp) for inp in node.inputs)})"
        return self._lbl(node)

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