"""Utilities to generate a graphical representation for a graph."""

from myia.anf_ir import Graph, Apply, Constant, Parameter
from myia.primops import Primitive
import os
import json


css_path = f'{os.path.dirname(__file__)}/graph.css'
css = open(css_path).read()


def is_computation(x):
    """Check if x is a computation."""
    return isinstance(x, Apply)


def is_constant(x):
    """Check if x is a constant."""
    return isinstance(x, Constant)


def is_parameter(x):
    """Check if x is a parameter."""
    return isinstance(x, Parameter)


def is_graph(x):
    """Check if x is a constant that contains a graph."""
    return isinstance(x, Constant) and isinstance(x.value, Graph)


class GraphPrinter:
    """
    Utility to generate a graphical representation for a graph.

    This is intended to be used as a base class for classes that
    specialize over particular graph structures.

    """

    def __init__(self, cyoptions):
        """Initialize GraphPrinter."""
        # Nodes and edges are accumulated in these lists
        self.nodes = []
        self.edges = []
        self.cyoptions = cyoptions

    def id(self, x):
        """Return the id associated to x."""
        return f'X{id(x)}'

    def fresh_id(self):
        """Return sequential identifier to guarantee a unique node."""
        self.currid += 1
        return f'Y{self.currid}'

    def cynode(self, id, label, classes, parent=None):
        """Build data structure for a node in cytoscape."""
        if not isinstance(id, str):
            id = self.id(id)
        data = {'id': id, 'label': str(label)}
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
        rval = H.cytoscapeGraph(H.style(css))
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
                 follow_references=False):
        """Initialize a MyiaGraphPrinter."""
        super().__init__({
            'layout': {
                'name': 'dagre',
                'rankDir': 'TB'
            }
        })
        # Graphs left to process
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.duplicate_free_variables = duplicate_free_variables
        self.function_in_node = function_in_node
        self.follow_references = follow_references
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
            'index': self.process_node_index,
            'make_tuple': self.process_node_make_tuple
        }

    def label_constant(self, ct):
        """Return a label for constants."""
        if isinstance(ct, Primitive):
            return ct.__class__.__name__.lower()
        else:
            return str(ct)

    def name(self, x):
        """Return the name of a node."""
        if x.debug.name:
            return x.debug.name
        else:
            return f'#{x.debug.id}'

    def const_fn(self, node):
        """
        Return name of function, if constant.

        Given an `Apply` node of a constant function, return the
        name of that function, otherwise return None.
        """
        fn = node.inputs[0] if node.inputs else None
        if fn and is_constant(fn):
            if is_graph(fn):
                return self.name(fn.value)
            elif isinstance(fn.value, Primitive):
                return fn.value.__class__.__name__.lower()
            else:
                return str(fn.value)
        else:
            return None

    def add_graph(self, g):
        """Create a node for a graph."""
        if g in self.processed:
            return
        name = self.name(g)
        argnames = [self.name(p) for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        self.cynode(id=g, label=lbl, classes='function')
        self.processed.add(g)

    def label(self, node, fn_label=None):
        """Return the label to give to a node."""
        if is_graph(node):
            lbl = self.name(node.value)
        elif is_constant(node):
            lbl = self.label_constant(node.value)
        elif is_parameter(node):
            lbl = self.name(node)
        else:
            lbl = ''
            if self.function_in_node:
                if fn_label is None:
                    fn_label = self.const_fn(node)
                if fn_label:
                    lbl = fn_label

            if node.debug.name:
                if lbl:
                    lbl += f'→{node.debug.name}'
                else:
                    lbl = node.debug.name
        return lbl or '·'

    def process_node_return(self, node, g, cl):
        """Create node and edges for `return ...`."""
        self.cynode(id=node, label='', parent=g, classes='const_output')
        ret = node.inputs[1]
        self.process_edges([(node, '', ret)])

    def process_node_index(self, node, g, cl):
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
        if fn and is_graph(fn):
            self.graphs.add(fn.value)

        edges = []
        if fn and not (is_constant(fn) and self.function_in_node):
            edges.append((node, 'F', fn))

        edges += [(node, i + 1, inp)
                  for i, inp in enumerate(node.inputs[1:]) or []]

        self.process_edges(edges)

    def process_node(self, node):
        """Create node and edges for a node."""
        if node in self.processed:
            return

        g = node.graph
        self.follow(node)

        if node in self.returns:
            cl = 'output'
        elif g and node in g.parameters:
            cl = 'input'
        elif is_constant(node):
            cl = 'constant'
        else:
            cl = 'intermediate'

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
                            classes='constant')
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
            elif self.duplicate_free_variables and \
                    src.graph and dest.graph and \
                    src.graph is not dest.graph:
                self.pool.add(dest)
                cid = self.fresh_id()
                self.cynode(id=cid,
                            parent=src.graph,
                            label=self.name(dest),
                            classes='freevar')
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
        if not is_computation(ret) or ret.graph is not g:
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
        if is_graph(node) and self.follow_references:
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
            return str(self)
        dc = hrepr.config.duplicate_constants
        dfv = hrepr.config.duplicate_free_variables
        fin = hrepr.config.function_in_node
        fr = hrepr.config.follow_references
        gpr = MyiaGraphPrinter(
            {self},
            duplicate_constants=True if dc is None else dc,
            duplicate_free_variables=True if dfv is None else dfv,
            function_in_node=True if fin is None else fin,
            follow_references=True if fr is None else fr
        )
        gpr.process()
        return gpr.__hrepr__(H, hrepr)
