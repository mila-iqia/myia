import os
from collections import defaultdict
from snektalk import pastevar


cystyle = open(os.path.join(os.path.dirname(__file__), "graph.css")).read()


class Graph:
    def __init__(self, edges, on_node=None):
        edges = list(edges)
        self.edges = edges
        self.nodes = {src for src, _ in edges} | {tgt for _, tgt in edges}
        self.pred = defaultdict(set)
        self.succ = defaultdict(set)
        for a, b in edges:
            self.pred[str(b)].add(a)
            self.succ[str(a)].add(b)
        self._on_node = on_node

    def on_node(self, data):
        if not self._on_node:
            return
        x = data["id"]
        data["pred"] = self.pred[x]
        data["succ"] = self.succ[x]
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
        width = hrepr.config.graph_width or 500
        height = hrepr.config.graph_height or 500
        style = hrepr.config.graph_style or cystyle
        data = [{"data": {"id": "P", "label": "Parent"}, "classes": "function"}]
        data += [
            {
                "data": {"id": node, "label": node, "parent": "P"},
                "classes": "constant",
            }
            for node in self.nodes
        ]
        data += [
            {"data": {"source": src, "target": tgt}} for src, tgt in self.edges
        ]
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


def main():
    print(Graph([(x // 2, x) for x in range(10)], on_node=pastevar))


if __name__ == "__main__":
    main()
