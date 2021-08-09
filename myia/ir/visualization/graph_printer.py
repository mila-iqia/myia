"""Wrapper for graph visualization in hrepr and snektalk."""
import os
from collections import deque

from myia.ir import Apply, Graph, Node
from myia.ir.node import FN, SEQ
from myia.ir.print import NodeLabeler
from myia.utils.misc import hrepr_include


class GraphPrinter:
    """Wrapper for graph visualization."""

    # CSS style for graph.
    __cystyle__ = open(
        os.path.join(os.path.dirname(__file__), "graph.css")
    ).read()

    def __init__(
        self,
        graph: Graph,
        *,
        on_node=None,
        show_fn_constants=False,
        show_args=False,
        link_fn_graphs=True,
        link_inp_graphs=True,
        labeler=None,
    ):
        """Initialize.

        Arguments:
            graph: graph to visualize
            on_node: optional function to call on printed node when clicked
            show_fn_constants: if True, display constant nodes when they are used
                in the FN edge of an Apply node.
            show_args: if True, display the list of arguments in shorthand inside
                an Apply node.
            link_fn_graphs: if True, display edges from apply FNs to constant graphs
            link_inp_graphs: if True, display edges from apply inputs to constant graphs
            labeler: NodeLabeler to use for labeling nodes and graphs
        """
        self.show_fn_constants = show_fn_constants
        self.show_args = show_args
        self.link_fn_graphs = link_fn_graphs
        self.link_inp_graphs = link_inp_graphs
        self.graphs, self.nodes, self.edges = self.collect_myia_elements(graph)
        self._on_node = on_node
        self._lbl = labeler or NodeLabeler()

    def on_node(self, data):  # pragma: no cover
        """Callback on given data when a node is clicked.

        Not called in tests. Needs snektalk to be tested.

        Arguments:
            data: dictionary representing clicked node.
                Contains at least "id" and "label" fields
        """
        if not self._on_node:
            return
        return self._on_node(self.id_to_element[data["id"]])

    @classmethod
    def __hrepr_resources__(cls, H):
        return [
            hrepr_include,
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
        # Generate identifiers to make hrepr output deterministic.
        elements = self.graphs + [n for _, n in self.nodes]
        identifiers = {
            element: str(index) for index, element in enumerate(elements)
        }
        self.id_to_element = {
            str(index): element for index, element in enumerate(elements)
        }
        assert len(identifiers) == len(elements)

        data = []

        # Graphs.
        data += [
            {
                "data": {
                    "id": identifiers[graph],
                    "label": self._lbl(graph),
                    "parent": identifiers.get(graph.parent, None),
                },
                "classes": "function",
            }
            for graph in self.graphs
        ]

        # Nodes.
        data += [
            {
                "data": {
                    "id": identifiers[node],
                    "label": self._lbl.informative(
                        node, show_args=self.show_args, hide_anonymous=True
                    ),
                    "parent": identifiers.get(g, None),
                },
                "classes": self.get_node_class(node),
            }
            for g, node in self.nodes
        ]

        # Edges.
        data += [
            {
                "data": {
                    "source": identifiers[tgt],
                    "target": identifiers[src],
                    "label": str(edge_label),
                },
                "classes": self.get_edge_class(edge_label),
            }
            for src, edge_label, tgt in self.edges
        ]

        width = hrepr.config.graph_width or 800
        height = hrepr.config.graph_height or 800
        style = hrepr.config.graph_style or self.__cystyle__
        return H.div["myia-GraphPrinter"](
            style=f"width:{width}px;height:{height}px;",
            constructor="make_graph",
            options={
                "elements": data,
                "style": style,
                "layout": {"name": "dagre"},
                "on_node": self.on_node,
            },
        )

    def collect_myia_elements(self, g: Graph):
        """Collect all elements to display.

        Arguments:
            g: graph to visualize

        Returns:
            A tuple:
            - list of myia graphs
            - list of myia nodes
            - list of edges.
              Each edge is a tuple (src element, edge label, tgt element).
              Element may be a graph or a node.
              Edge label is label field from myia Edge object.
        """
        all_graphs = []
        all_nodes = []
        all_edges = []
        seen_graphs = set()
        seen_nodes = set()
        seen_edges = set()
        todo_graphs = deque([g])
        while todo_graphs:
            graph = todo_graphs.popleft()
            if graph in seen_graphs:
                continue
            seen_graphs.add(graph)
            all_graphs.append(graph)
            # Register parameters immediately, so that they will be labeled before other graph nodes.
            for p in graph.parameters:
                if p not in seen_nodes:
                    seen_nodes.add(p)
                    all_nodes.append((p.graph, p))
            todo_edges = deque([(None, None, graph.return_)])
            while todo_edges:
                edge = todo_edges.popleft()
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                user, label, used = edge
                if user is not None:
                    all_edges.append(edge)
                if isinstance(used, Node) and used not in seen_nodes:
                    seen_nodes.add(used)
                    all_nodes.append(
                        (getattr(used, "graph", None) or user.graph, used)
                    )
                if isinstance(used, Apply):
                    inputs = set(used.inputs)
                    for e in used.edges.values():
                        node = e.node
                        if node.is_constant_graph():
                            todo_graphs.append(node.value)
                            if (self.link_fn_graphs and node is used.fn) or (
                                self.link_inp_graphs and node in inputs
                            ):
                                todo_edges.append((used, e.label, node.value))
                            else:
                                todo_edges.append((used, e.label, node))
                        elif (
                            self.show_fn_constants
                            or e.label is not FN
                            or not node.is_constant()
                        ):
                            todo_edges.append((used, e.label, node))
        return all_graphs, all_nodes, all_edges

    def get_node_class(self, node):
        """Get CSS class for given node."""
        if node.is_parameter():
            return "input"
        if node.is_constant():
            return "constant"
        if node is node.graph.return_:
            return "return"
        return "intermediate"

    def get_edge_class(self, edge_label):
        """Get CSS class for given edge label."""
        assert edge_label in (SEQ, FN) or isinstance(edge_label, (int, str))
        if edge_label is SEQ:
            return "link-edge"
        if edge_label is FN:
            return "fn-edge"
        if isinstance(edge_label, (int, str)):
            return "input-edge"
