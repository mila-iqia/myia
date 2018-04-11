"""Utilities to generate a graphical representation for a graph."""

from hrepr import hrepr
import os
import json

from myia.anf_ir import Graph, ANFNode, Apply, Constant
from myia.anf_ir_utils import \
    is_apply, is_constant, is_constant_graph
from myia.parser import Location
from myia import primops
from myia.cconv import NestingAnalyzer, ParentProxy
from myia.debug.label import NodeLabeler, short_relation_symbols, short_labeler


gcss_path = f'{os.path.dirname(__file__)}/graph.css'
gcss = open(gcss_path).read()

mcss_path = f'{os.path.dirname(__file__)}/myia.css'
mcss = open(mcss_path).read()


class GraphPrinter:
    """Utility to generate a graphical representation for a graph.

    This is intended to be used as a base class for classes that
    specialize over particular graph structures.

    """

    def __init__(self,
                 cyoptions,
                 tooltip_gen=None,
                 extra_style=None):
        """Initialize GraphPrinter."""
        # Nodes and edges are accumulated in these lists
        self.nodes = []
        self.edges = []
        self.cyoptions = cyoptions
        self.tooltip_gen = tooltip_gen
        self.extra_style = extra_style or ''

    def id(self, x):
        """Return the id associated to x."""
        return f'X{id(x)}'

    def fresh_id(self):
        """Return sequential identifier to guarantee a unique node."""
        self.currid += 1
        return f'Y{self.currid}'

    def cynode(self, id, label, classes, parent=None, node=None):
        """Build data structure for a node in cytoscape."""
        if not isinstance(id, str):
            if node is None:
                node = id
            id = self.id(id)
        data = {'id': id, 'label': str(label)}
        if self.tooltip_gen and node:
            ttip = self.tooltip_gen(node)
            if ttip is not None:
                if not isinstance(ttip, str):
                    ttip = str(hrepr(ttip))
                data['tooltip'] = ttip
        if parent:
            parent = parent if isinstance(parent, str) else self.id(parent)
            data['parent'] = parent
        self.nodes.append({'data': data, 'classes': classes})

    def cyedge(self, src_id, dest_id, label):
        """Build data structure for an edge in cytoscape."""
        cl = 'input-edge'
        if isinstance(label, tuple):
            label, cl = label
        if not isinstance(label, str):
            label = str(label)
        if not isinstance(dest_id, str):
            dest_id = self.id(dest_id)
        if not isinstance(src_id, str):
            src_id = self.id(src_id)
        data = {
            'id': f'{dest_id}-{src_id}-{label}',
            'label': label,
            'source': dest_id,
            'target': src_id
        }
        self.edges.append({'data': data, 'classes': cl})

    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return H.bucheRequire(name='cytoscape',
                              channels='cytoscape',
                              components='cytoscape-graph')

    def __hrepr__(self, H, hrepr):
        """Return HTML representation (uses buche-cytoscape)."""
        rval = H.cytoscapeGraph(H.style(gcss + self.extra_style))
        rval = rval(width=hrepr.config.graph_width or '800px',
                    height=hrepr.config.graph_height or '500px')
        rval = rval(H.options(json.dumps(self.cyoptions)))
        for elem in self.nodes + self.edges:
            rval = rval(H.element(json.dumps(elem)))
        return rval


class MyiaGraphPrinter(GraphPrinter):
    """
    Utility to generate a graphical representation for a graph.

    Attributes:
        duplicate_constants: Whether to create a separate node for
            every instance of the same constant.
        duplicate_free_variables: Whether to create a separate node
            to represent the use of a free variable, or point directly
            to that node in a different graph.
        function_in_node: Whether to print, when possible, the name
            of a node's operation directly in the node's label instead
            of creating a node for the operation and drawing an edge
            to it.
        follow_references: Whether to also print graphs that are
            called by this graph.

    """

    def __init__(self,
                 entry_points,
                 *,
                 duplicate_constants=False,
                 duplicate_free_variables=False,
                 function_in_node=False,
                 follow_references=False,
                 tooltip_gen=None,
                 class_gen=None,
                 extra_style=None):
        """Initialize a MyiaGraphPrinter."""
        super().__init__(
            {
                'layout': {
                    'name': 'dagre',
                    'rankDir': 'TB'
                }
            },
            tooltip_gen=tooltip_gen,
            extra_style=extra_style
        )
        # Graphs left to process
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.duplicate_free_variables = duplicate_free_variables
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        self.labeler = NodeLabeler(
            function_in_node=function_in_node,
            relation_symbols=short_relation_symbols
        )
        self._class_gen = class_gen
        # Nodes processed
        self.processed = set()
        # Nodes left to process
        self.pool = set()
        # Nodes that are to be colored as return nodes
        self.returns = set()
        # IDs for duplicated constants
        self.currid = 0
        # Custom rules for nodes that represent certain calls
        self.custom_rules = {
            'return': self.process_node_return,
            'getitem': self.process_node_getitem,
            'make_tuple': self.process_node_make_tuple
        }

    def name(self, x):
        """Return the name of a node."""
        return self.labeler.name(x, force=True)

    def label(self, node, fn_label=None):
        """Return the label to give to a node."""
        return self.labeler.label(node, None, fn_label=fn_label)

    def const_fn(self, node):
        """
        Return name of function, if constant.

        Given an `Apply` node of a constant function, return the
        name of that function, otherwise return None.
        """
        return self.labeler.const_fn(node)

    def add_graph(self, g):
        """Create a node for a graph."""
        if g in self.processed:
            return
        name = self.name(g)
        argnames = [self.name(p) for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        self.cynode(id=g, label=lbl, classes='function')
        self.processed.add(g)

    def process_node_return(self, node, g, cl):
        """Create node and edges for `return ...`."""
        self.cynode(id=node, label='', parent=g, classes='const_output')
        ret = node.inputs[1]
        self.process_edges([(node, '', ret)])

    def process_node_getitem(self, node, g, cl):
        """Create node and edges for `x[ct]`."""
        idx = node.inputs[2]
        if self.function_in_node and is_constant(idx):
            lbl = self.label(node, '')
            self.cynode(id=node, label=lbl, parent=g, classes=cl)
            self.process_edges([(node,
                                 (f'[{idx.value}]', 'fn-edge'),
                                 node.inputs[1])])
        else:
            self.process_node_generic(node, g, cl)

    def process_node_make_tuple(self, node, g, cl):
        """Create node and edges for `(a, b, c, ...)`."""
        if self.function_in_node:
            lbl = self.label(node, f'(...)')
            self.cynode(id=node, label=lbl, parent=g, classes=cl)
            edges = [(node, i + 1, inp)
                     for i, inp in enumerate(node.inputs[1:]) or []]
            self.process_edges(edges)
        else:
            return self.process_node_generic(node, g, cl)

    def process_node_generic(self, node, g, cl):
        """Create node and edges for a node."""
        lbl = self.label(node)

        self.cynode(id=node, label=lbl, parent=g, classes=cl)

        fn = node.inputs[0] if node.inputs else None
        if fn and is_constant_graph(fn):
            self.graphs.add(fn.value)

        edges = []
        if fn and not (is_constant(fn) and self.function_in_node):
            edges.append((node, 'F', fn))

        edges += [(node, i + 1, inp)
                  for i, inp in enumerate(node.inputs[1:]) or []]

        self.process_edges(edges)

    def class_gen(self, node, cl=None):
        """Generate the class name for this node."""
        g = node.graph
        if cl is not None:
            pass
        elif node in self.returns:
            cl = 'output'
        elif g and node in g.parameters:
            cl = 'input'
        elif is_constant(node):
            cl = 'constant'
        else:
            cl = 'intermediate'
        if self._class_gen:
            return self._class_gen(node, cl)
        else:
            return cl

    def process_node(self, node):
        """Create node and edges for a node."""
        if node in self.processed:
            return

        g = node.graph
        self.follow(node)
        cl = self.class_gen(node)

        ctfn = self.const_fn(node)
        if ctfn:
            if ctfn in self.custom_rules:
                self.custom_rules[ctfn](node, g, cl)
            else:
                self.process_node_generic(node, g, cl)
        else:
            self.process_node_generic(node, g, cl)

        self.processed.add(node)

    def process_edges(self, edges):
        """Create edges."""
        for edge in edges:
            src, lbl, dest = edge
            if is_constant(dest) and self.duplicate_constants:
                self.follow(dest)
                cid = self.fresh_id()
                self.cynode(id=cid,
                            parent=src.graph,
                            label=self.label(dest),
                            classes=self.class_gen(dest, 'constant'),
                            node=dest)
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
            elif self.duplicate_free_variables and \
                    src.graph and dest.graph and \
                    src.graph is not dest.graph:
                self.pool.add(dest)
                cid = self.fresh_id()
                self.cynode(id=cid,
                            parent=src.graph,
                            label=self.name(dest),
                            classes=self.class_gen(dest, 'freevar'),
                            node=dest)
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
                self.cyedge(src_id=cid, dest_id=dest, label=(lbl, 'link-edge'))
            else:
                self.pool.add(dest)
                self.cyedge(src_id=src, dest_id=dest, label=lbl)

    def process_graph(self, g):
        """Process a graph."""
        self.add_graph(g)
        for inp in g.parameters:
            self.process_node(inp)

        if not g.return_:
            return

        ret = g.return_.inputs[1]
        if not is_apply(ret) or ret.graph is not g:
            ret = g.return_

        self.returns.add(ret)
        self.pool.add(ret)

        while self.pool:
            node = self.pool.pop()
            self.process_node(node)

    def process(self):
        """Process all graphs in entry_points."""
        if self.nodes or self.edges:
            return
        while self.graphs:
            g = self.graphs.pop()
            self.process_graph(g)
        return self.nodes, self.edges

    def follow(self, node):
        """Add this node's graph if follow_references is True."""
        if is_constant_graph(node) and self.follow_references:
            self.graphs.add(node.value)


class Empty:
    """Bogus class."""

    pass


def mixin(target):
    """Class decorator to add methods to the target class."""
    def apply(cls):
        methods = set(dir(cls))
        methods.difference_update(set(dir(Empty)))
        for mthd in methods:
            setattr(target, mthd, getattr(cls, mthd))
        return target
    return apply


@mixin(Graph)
class _Graph:
    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return GraphPrinter.__hrepr_resources__(H)

    def __hrepr__(self, H, hrepr):
        """Return HTML representation (uses buche-cytoscape)."""
        if hrepr.config.depth > 1 and not hrepr.config.graph_expand_all:
            return hrepr(Constant(self))
        dc = hrepr.config.duplicate_constants
        dfv = hrepr.config.duplicate_free_variables
        fin = hrepr.config.function_in_node
        fr = hrepr.config.follow_references
        tgen = hrepr.config.node_tooltip
        cgen = hrepr.config.node_class
        xsty = hrepr.config.graph_style
        gpr = MyiaGraphPrinter(
            {self},
            duplicate_constants=True if dc is None else dc,
            duplicate_free_variables=True if dfv is None else dfv,
            function_in_node=True if fin is None else fin,
            follow_references=True if fr is None else fr,
            tooltip_gen=tgen,
            class_gen=cgen,
            extra_style=xsty
        )
        gpr.process()
        return gpr.__hrepr__(H, hrepr)


@mixin(ANFNode)
class _ANFNode:
    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return H.style(mcss)

    def __hrepr__(self, H, hrepr):
        class_name = self.__class__.__name__.lower()
        label = short_labeler.label(self, True)
        return H.span['node', f'node-{class_name}'](label)


@mixin(Apply)
class _Apply:
    def __hrepr__(self, H, hrepr):
        if len(self.inputs) == 2 and \
                isinstance(self.inputs[0], Constant) and \
                self.inputs[0].value is primops.return_:
            return hrepr.hrepr_nowrap(self.inputs[1])['node-return']
        else:
            return super(Apply, self).__hrepr__(H, hrepr)


@mixin(Location)
class _Location:
    def __hrepr__(self, H, hrepr):
        return H.div(
            H.style('.hljs, .hljs-linenos { font-size: 10px !IMPORTANT; }'),
            H.codeSnippet(
                src=self.url,
                language="python",
                line=self.line,
                column=self.column + 1,
                context=hrepr.config.snippet_context or 4
            )
        )


@mixin(NestingAnalyzer)
class _NestingAnalyzer:
    @classmethod
    def __hrepr_resources__(cls, H):
        return GraphPrinter.__hrepr_resources__(H)

    def __hrepr__(self, H, hrepr):
        pr = GraphPrinter({
            'layout': {
                'name': 'dagre',
                'rankDir': 'TB'
            }
        })

        mode = hrepr.config.nesting_analyzer_mode or 'parents'

        def lbl(x):
            if isinstance(x, ParentProxy):
                return f"{x.graph.debug.debug_name}'"
            else:
                return x.debug.debug_name

        if mode == 'parents':
            graph = {g: set() if parent is None else {parent}
                     for g, parent in self.parents().items()}
        elif mode == 'children':
            graph = self.children()
        elif mode == 'graph_dependencies_direct':
            graph = self.graph_dependencies_direct()
        elif mode == 'graph_dependencies_total':
            graph = self.graph_dependencies_total()
        else:
            raise Exception(
                f'Unknown display mode for NestingAnalyzer: "{mode}"'
            )

        for g, deps in graph.items():
            pr.cynode(g, lbl(g), 'intermediate')
            for dep in deps:
                pr.cynode(dep, lbl(dep), 'intermediate')
                pr.cynode(dep, lbl(dep), 'intermediate')
                pr.cyedge(g, dep, '')
        return pr.__hrepr__(H, hrepr)
