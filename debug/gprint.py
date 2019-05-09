"""Utilities to generate a graphical representation for a graph."""

import json
import os

from hrepr import hrepr

from myia.abstract import (
    AbstractValue, AbstractScalar, AbstractFunction, AbstractTuple,
    AbstractList, AbstractClass, AbstractJTagged, AbstractArray,
    GraphFunction, PartialApplication, TypedPrimitive, PrimitiveFunction,
    MetaGraphFunction, AbstractUnion, ConditionalContext, VALUE, ANYTHING,
    PendingTentative
)
from myia.dtype import Type, Bool, Int, Float, TypeMeta, UInt
from myia.utils import OrderedSet, UNKNOWN

try:
    from myia.dtype import JTagged
except ImportError:
    class JTagged:
        pass

from myia.abstract import Reference, VirtualReference, Context
from myia.info import DebugInfo, About
from myia.ir import ANFNode, Apply, Constant, Graph, GraphCloner, \
    ParentProxy, GraphManager, manage
from myia.parser import Location
from myia.prim import ops as primops
from myia.opt import LocalPassOptimizer, pattern_replacer, NodeMap
from myia.utils import Registry, NS
from myia.utils.unify import Var, SVar, var, FilterVar
from myia.vm import VMFrame, Closure

from myia.debug.label import NodeLabeler, short_labeler, \
    short_relation_symbols, CosmeticPrimitive
from myia.debug.utils import mixin


gcss_path = f'{os.path.dirname(__file__)}/graph.css'
gcss = open(gcss_path).read()

mcss_path = f'{os.path.dirname(__file__)}/myia.css'
mcss = open(mcss_path).read()


def _has_error(dbg):
    # Whether an error occurred somewhere in a DebugInfo
    if dbg.errors:
        return True
    elif dbg.about:
        return _has_error(dbg.about.debug)
    else:
        return False


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

    def _strip_cosmetic(self, node):
        while node and node.debug.about \
                and node.debug.about.relation == 'cosmetic':
            node = node.debug.about.debug.obj
        return node

    def cynode(self, id, label, classes, parent=None, node=None):
        """Build data structure for a node in cytoscape."""
        if not isinstance(id, str):
            if node is None:
                node = id
            id = self.id(id)
        data = {'id': id, 'label': str(label)}
        if self.tooltip_gen and node:
            ttip = self.tooltip_gen(self._strip_cosmetic(node))
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
        return (H.style(mcss),
                H.bucheRequire(name='cytoscape',
                               channels='cytoscape',
                               components='cytoscape-graph'))

    def __hrepr__(self, H, hrepr):
        """Return HTML representation (uses buche-cytoscape)."""
        opts = {
            'style': gcss + self.extra_style,
            'elements': self.nodes + self.edges,
        }
        return H.cytoscapeGraph(
            H.script(
                json.dumps({**opts, **self.cyoptions}),
                type='buche/configure'
            ),
            width=hrepr.config.graph_width or '800px',
            height=hrepr.config.graph_height or '500px',
            interactive=True,
        )


def _make_class_gen(cgen):
    if isinstance(cgen, (tuple, list, set, frozenset)):
        cgen = frozenset(cgen)
        return lambda x, cl: f'error {cl}' if x in cgen else cl
    elif isinstance(cgen, dict):
        return lambda x, cl: f'{cgen[x]} {cl}' if x in cgen else cl
    else:
        return cgen


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
                 extra_style=None,
                 beautify=True):
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
        if beautify:
            self.graphs = set()
            self.focus = set()
            for g in entry_points:
                self._import_graph(g)
        else:
            self.graphs = set(entry_points)
            self.focus = set(self.graphs)

        self.beautify = beautify
        self.duplicate_constants = duplicate_constants
        self.duplicate_free_variables = duplicate_free_variables
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        self.labeler = NodeLabeler(
            function_in_node=function_in_node,
            relation_symbols=short_relation_symbols
        )
        self._class_gen = _make_class_gen(class_gen)
        # Nodes processed
        self.processed = set()
        # Nodes left to process
        self.pool = set()
        # Nodes that are to be colored as return nodes
        self.returns = set()
        # IDs for duplicated constants
        self.currid = 0

    def _import_graph(self, graph):
        mng = manage(graph, weak=True)
        graphs = set()
        parents = mng.parents
        g = graph
        while g:
            graphs.add(g)
            g = parents[g]
        clone = GraphCloner(*graphs, total=True, relation='cosmetic')
        self.graphs |= {clone[g] for g in graphs}
        self.focus.add(clone[graph])

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
        if self.beautify:
            g = cosmetic_transformer(g)
        name = self.name(g)
        argnames = [self.name(p) for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        classes = ['function', 'focus' if g in self.focus else '']
        self.cynode(id=g, label=lbl, classes=' '.join(classes))
        self.processed.add(g)

    def process_node_generic(self, node, g, cl):
        """Create node and edges for a node."""
        lbl = self.label(node)

        self.cynode(id=node, label=lbl, parent=g, classes=cl)

        fn = node.inputs[0] if node.inputs else None
        if fn and fn.is_constant_graph():
            self.graphs.add(fn.value)

        for inp in node.inputs:
            if inp.is_constant_graph():
                self.cyedge(src_id=g, dest_id=inp.value,
                            label=('', 'use-edge'))

        edges = []
        if fn and not (fn.is_constant() and self.function_in_node):
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
        elif node.is_parameter():
            cl = 'input'
            if node not in g.parameters:
                cl += ' unlisted'
        elif node.is_constant():
            cl = 'constant'
        elif node.is_special():
            cl = f'special-{type(node.special).__name__}'
        else:
            cl = 'intermediate'
        if _has_error(node.debug):
            cl += ' error'
        if self._class_gen:
            return self._class_gen(self._strip_cosmetic(node), cl)
        else:
            return cl

    def process_node(self, node):
        """Create node and edges for a node."""
        if node in self.processed:
            return

        g = node.graph
        self.follow(node)
        cl = self.class_gen(node)
        if g and g not in self.processed:
            self.add_graph(g)

        if node.inputs and node.inputs[0].is_constant():
            fn = node.inputs[0].value
            if fn in cosmetics:
                cosmetics[fn](self, node, g, cl)
            elif hasattr(fn, 'graph_display'):
                fn.graph_display(self, node, g, cl)
            else:
                self.process_node_generic(node, g, cl)
        else:
            self.process_node_generic(node, g, cl)

        self.processed.add(node)

    def process_edges(self, edges):
        """Create edges."""
        for edge in edges:
            src, lbl, dest = edge
            if dest.is_constant() and self.duplicate_constants:
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
                            label=self.labeler.label(dest, force=True),
                            classes=self.class_gen(dest, 'freevar'),
                            node=dest)
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
                self.cyedge(src_id=cid, dest_id=dest, label=(lbl, 'link-edge'))
                self.cyedge(src_id=src.graph, dest_id=dest.graph,
                            label=('', 'nest-edge'))
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
        if not ret.is_apply() or ret.graph is not g:
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
        if node.is_constant_graph() and self.follow_references:
            self.graphs.add(node.value)


class MyiaNodesPrinter(GraphPrinter):

    def __init__(self,
                 nodes,
                 *,
                 duplicate_constants=True,
                 duplicate_free_variables=True,
                 function_in_node=True,
                 tooltip_gen=None,
                 class_gen=None,
                 extra_style=None):
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
        self.duplicate_constants = duplicate_constants
        self.duplicate_free_variables = duplicate_free_variables
        self.function_in_node = function_in_node
        self.labeler = NodeLabeler(
            function_in_node=function_in_node,
            relation_symbols=short_relation_symbols
        )
        self._class_gen = _make_class_gen(class_gen)
        self.todo = set(nodes)
        self.graphs = {node.graph for node in nodes if node.graph}
        self.focus = set()
        # Nodes that are to be colored as return nodes
        self.returns = {node for node in nodes
                        if node.graph and node is node.graph.return_}
        # IDs for duplicated constants
        self.currid = 0

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
        name = self.name(g)
        argnames = [self.name(p) for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        classes = ['function', 'focus' if g in self.focus else '']
        self.cynode(id=g, label=lbl, classes=' '.join(classes))
        # self.processed.add(g)

    def process_node_generic(self, node, g, cl):
        """Create node and edges for a node."""
        if node.is_constant() and self.duplicate_constants:
            return

        lbl = self.label(node)

        self.cynode(id=node, label=lbl, parent=g, classes=cl)

        fn = node.inputs[0] if node.inputs else None
        if fn and fn.is_constant_graph():
            self.graphs.add(fn.value)

        for inp in node.inputs:
            if inp.is_constant_graph():
                self.cyedge(src_id=g, dest_id=inp.value,
                            label=('', 'use-edge'))

        edges = []
        if fn and not (fn.is_constant() and self.function_in_node):
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
        elif node.is_parameter():
            cl = 'input'
            if node not in g.parameters:
                cl += ' unlisted'
        elif node.is_constant():
            cl = 'constant'
        elif node.is_special():
            cl = f'special-{type(node.special).__name__}'
        else:
            cl = 'intermediate'
        if _has_error(node.debug):
            cl += ' error'
        if self._class_gen:
            return self._class_gen(self._strip_cosmetic(node), cl)
        else:
            return cl

    def process_node(self, node):
        """Create node and edges for a node."""
        # if node in self.processed:
        #     return

        g = node.graph
        # self.follow(node)
        cl = self.class_gen(node)

        if node.inputs and node.inputs[0].is_constant():
            fn = node.inputs[0].value
            if fn in cosmetics:
                cosmetics[fn](self, node, g, cl)
            elif hasattr(fn, 'graph_display'):
                fn.graph_display(self, node, g, cl)
            else:
                self.process_node_generic(node, g, cl)
        else:
            self.process_node_generic(node, g, cl)

    def process_edges(self, edges):
        """Create edges."""
        for edge in edges:
            src, lbl, dest = edge
            if dest not in self.todo:
                continue
            if dest.is_constant() and self.duplicate_constants:
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
                cid = self.fresh_id()
                self.cynode(id=cid,
                            parent=src.graph,
                            label=self.labeler.label(dest, force=True),
                            classes=self.class_gen(dest, 'freevar'),
                            node=dest)
                self.cyedge(src_id=src, dest_id=cid, label=lbl)
                self.cyedge(src_id=cid, dest_id=dest, label=(lbl, 'link-edge'))
                self.cyedge(src_id=src.graph, dest_id=dest.graph,
                            label=('', 'nest-edge'))
            else:
                self.cyedge(src_id=src, dest_id=dest, label=lbl)

    def process(self):
        """Process all graphs in entry_points."""
        if self.nodes or self.edges:
            return
        for g in self.graphs:
            self.add_graph(g)
        for node in self.todo:
            self.process_node(node)
        return self.nodes, self.edges


cosmetics = Registry()


@cosmetics.register(primops.return_)
def _cosmetic_node_return(self, node, g, cl):
    """Create node and edges for `return ...`."""
    self.cynode(id=node, label='', parent=g, classes='const_output')
    ret = node.inputs[1]
    self.process_edges([(node, '', ret)])


class GraphCosmeticPrimitive(CosmeticPrimitive):
    """Cosmetic primitive that prints pretty in graphs.

    Attributes:
        on_edge: Whether to display the label on the edge.

    """

    def __init__(self, label, on_edge=False):
        """Initialize a GraphCosmeticPrimitive."""
        super().__init__(label)
        self.on_edge = on_edge

    def graph_display(self, gprint, node, g, cl):
        """Display a node in cytoscape graph."""
        if gprint.function_in_node and self.on_edge:
            lbl = gprint.label(node, '')
            gprint.cynode(id=node, label=lbl, parent=g, classes=cl)
            gprint.process_edges([(node,
                                   (self.label, 'fn-edge'),
                                   node.inputs[1])])
        else:
            gprint.process_node_generic(node, g, cl)


make_tuple = GraphCosmeticPrimitive('(...)')


X = Var('X')
Y = Var('Y')
Xs = SVar(Var())
V = var(lambda x: x.is_constant())
V1 = var(lambda x: x.is_constant())
V2 = var(lambda x: x.is_constant())
L = var(lambda x: x.is_constant_graph())


@pattern_replacer(primops.make_tuple, Xs)
def _opt_fancy_make_tuple(optimizer, node, equiv):
    xs = equiv[Xs]
    ct = Constant(GraphCosmeticPrimitive('(...)'))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, *xs], node.graph)


@pattern_replacer(primops.tuple_getitem, X, V)
def _opt_fancy_getitem(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f'[{v.value}]', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.resolve, V1, V2)
def _opt_fancy_resolve(optimizer, node, equiv):
    ns = equiv[V1]
    name = equiv[V2]
    with About(node.debug, 'cosmetic'):
        lbl = f'{ns.value.label}.{name.value}'
        ct = Constant(GraphCosmeticPrimitive(lbl))
        return ct


@pattern_replacer(primops.getattr, X, V)
def _opt_fancy_getattr(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f'{v.value}', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.unsafe_static_cast, X, V)
def _opt_fancy_unsafe_static_cast(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f'cast', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.array_map, V, Xs)
def _opt_fancy_array_map(optimizer, node, equiv):
    xs = equiv[Xs]
    v = equiv[V]
    if v.is_constant_graph():
        return node
    name = short_labeler.label(v)
    ct = Constant(GraphCosmeticPrimitive(f'[{name}]'))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, *xs], node.graph)


@pattern_replacer(primops.distribute, X, V)
def _opt_fancy_distribute(optimizer, node, equiv):
    x = equiv[X]
    v = equiv[V]
    ct = Constant(GraphCosmeticPrimitive(f'shape→{v.value}', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.scalar_to_array, X)
def _opt_fancy_scalar_to_array(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f'to_array', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.array_to_scalar, X)
def _opt_fancy_array_to_scalar(optimizer, node, equiv):
    x = equiv[X]
    ct = Constant(GraphCosmeticPrimitive(f'to_scalar', on_edge=True))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.transpose, X, V)
def _opt_fancy_transpose(optimizer, node, equiv):
    if equiv[V].value == (1, 0):
        x = equiv[X]
        ct = Constant(GraphCosmeticPrimitive(f'T', on_edge=True))
        with About(node.debug, 'cosmetic'):
            return Apply([ct, x], node.graph)
    else:
        return node


@pattern_replacer(primops.array_reduce, primops.scalar_add, X, V)
def _opt_fancy_sum(optimizer, node, equiv):
    x = equiv[X]
    shp = equiv[V].value
    ct = Constant(GraphCosmeticPrimitive(f'sum {"x".join(map(str, shp))}'))
    with About(node.debug, 'cosmetic'):
        return Apply([ct, x], node.graph)


@pattern_replacer(primops.distribute, (primops.scalar_to_array, V), X)
def _opt_distributed_constant(optimizer, node, equiv):
    return equiv[V]


def cosmetic_transformer(g):
    """Transform a graph so that it looks nicer.

    The resulting graph is not a valid one to run, because it may contain nodes
    with fake functions that only serve a cosmetic purpose.
    """
    spec = (
        _opt_distributed_constant,
        _opt_fancy_make_tuple,
        _opt_fancy_getitem,
        _opt_fancy_resolve,
        _opt_fancy_getattr,
        _opt_fancy_array_map,
        _opt_fancy_distribute,
        _opt_fancy_transpose,
        _opt_fancy_sum,
        _opt_fancy_unsafe_static_cast,
        # _opt_fancy_scalar_to_array,
        _opt_fancy_array_to_scalar,
        # careful=True
    )
    nmap = NodeMap()
    for opt in spec:
        nmap.register(getattr(opt, 'interest', None), opt)
    opt = LocalPassOptimizer(nmap)
    opt(g)
    return g


@mixin(Graph)
class _Graph:
    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return GraphPrinter.__hrepr_resources__(H)

    def __hrepr__(self, H, hrepr):
        """Return HTML representation (uses buche-cytoscape)."""
        if hrepr.config.depth > 1 and not hrepr.config.graph_expand_all:
            label = short_labeler.label(self, True)
            return H.span['node', f'node-Graph'](label)
        dc = hrepr.config.duplicate_constants
        dfv = hrepr.config.duplicate_free_variables
        fin = hrepr.config.function_in_node
        fr = hrepr.config.follow_references
        tgen = hrepr.config.node_tooltip
        cgen = hrepr.config.node_class
        xsty = hrepr.config.graph_style
        beau = hrepr.config.graph_beautify
        gpr = MyiaGraphPrinter(
            {self},
            duplicate_constants=True if dc is None else dc,
            duplicate_free_variables=True if dfv is None else dfv,
            function_in_node=True if fin is None else fin,
            follow_references=True if fr is None else fr,
            tooltip_gen=tgen,
            class_gen=cgen,
            extra_style=xsty,
            beautify=True if beau is None else beau
        )
        gpr.process()
        return gpr.__hrepr__(H, hrepr)


#################
# Node printers #
#################


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
            if hasattr(hrepr, 'hrepr_nowrap'):
                return hrepr.hrepr_nowrap(self.inputs[1])['node-return']
            else:
                return hrepr(self.inputs[1])['node-return']
        else:
            return super(Apply, self).__hrepr__(H, hrepr)


@mixin(ParentProxy)
class _ParentProxy:
    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return H.style(mcss)

    def __hrepr__(self, H, hrepr):
        class_name = 'constant'
        label = 'Prox:' + short_labeler.label(self.graph, True)
        return H.span['node', f'node-{class_name}'](label)


########
# Misc #
########


@mixin(NS)
class _NS:
    def __hrepr__(self, H, hrepr):
        return hrepr(self.__dict__)


@mixin(OrderedSet)
class _OrderedSet:
    def __hrepr__(self, H, hrepr):
        return hrepr(set(self._d.keys()))


@mixin(VirtualReference)
class _VirtualReference:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'VirtualReference',
            self.values.items()
        )


@mixin(Reference)
class _Reference:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'Reference',
            (('node', self.node), ('context', self.context))
        )


@mixin(Context)
class _Context:
    def __hrepr__(self, H, hrepr):
        stack = []
        curr = self
        while curr:
            stack.append((curr.graph, curr.argkey))
            curr = curr.parent
        return hrepr.stdrepr_object('Context', stack)


@mixin(ConditionalContext)
class _ConditionalContext:
    def __hrepr__(self, H, hrepr):
        stack = []
        curr = self
        while curr:
            stack.append((curr.graph, curr.argkey))
            curr = curr.parent
        return hrepr.stdrepr_object('ConditionalContext', stack)


@mixin(Location)
class _Location:
    def __hrepr__(self, H, hrepr):
        return H.div(
            H.style('.hljs, .hljs-linenos { font-size: 10px !IMPORTANT; }'),
            H.codeSnippet(
                src=self.filename,
                language="python",
                line=self.line,
                column=self.column + 1,
                context=hrepr.config.snippet_context or 4
            )
        )


@mixin(DebugInfo)
class _DebugInfo:
    def __hrepr__(self, H, hrepr):
        exclude = {'save_trace', 'about'}

        def mkdict(info):
            d = {k: v for k, v in info.__dict__.items()
                 if not k.startswith('_') and k not in exclude}
            tr = d.get('trace', None)
            if tr:
                fr = tr[-3]
                d['trace'] = Location(
                    fr.filename, fr.lineno, 0, fr.lineno, 0, None
                )
            d['name'] = short_labeler.label(info)
            return d

        tabs = []
        info = self
        while getattr(info, 'about', None):
            tabs.append((info, info.about.relation))
            info = info.about.debug
        tabs.append((info, 'initial'))

        rval = H.boxTabs()
        for info, rel in tabs:
            pane = hrepr(mkdict(info))
            rval = rval(H.tabEntry(H.tabLabel(rel), H.tabPane(pane)))

        return rval


@mixin(GraphManager)
class _GraphManager:
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

        mode = hrepr.config.manager_mode or 'parents'

        def lbl(x):
            if isinstance(x, ParentProxy):
                return f"{short_labeler.label(x.graph)}'"
            else:
                return short_labeler.label(x)

        if mode == 'parents':
            graph = {g: set() if parent is None else {parent}
                     for g, parent in self.parents.items()}
        elif mode == 'children':
            graph = self.children
        elif mode == 'graph_dependencies_direct':
            graph = self.graph_dependencies_direct
        elif mode == 'graph_dependencies_total':
            graph = self.graph_dependencies_total
        else:
            raise Exception(
                f'Unknown display mode for GraphManager: "{mode}"'
            )

        for g, deps in graph.items():
            pr.cynode(g, lbl(g), 'intermediate')
            for dep in deps:
                pr.cynode(dep, lbl(dep), 'intermediate')
                pr.cynode(dep, lbl(dep), 'intermediate')
                pr.cyedge(g, dep, '')
        return pr.__hrepr__(H, hrepr)


@mixin(VMFrame)
class _VMFrame:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'Frame',
            [
                ('graph', self.graph),
                ('values', self.values),
                ('closure', self.closure),
                ('todo', self.todo),
                ('ownership', [x.graph in (None, self.graph)
                               for x in self.todo])
            ]
        )


@mixin(Closure)
class _Closure:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object('Closure', [
            ('graph', self.graph),
            ('values', self.values)
        ])


#################
# Type printers #
#################


@mixin(TypeMeta)
class _TypeMeta:
    def __hrepr__(cls, H, hrepr):
        return cls.__type_hrepr__(H, hrepr)

    @classmethod
    def __hrepr_resources__(cls, H):
        return H.style(mcss)


@mixin(Type)
class _Type:
    @classmethod
    def __type_hrepr__(cls, H, hrepr):
        return H.span(str(cls))


@mixin(Bool)
class _Bool:
    @classmethod
    def __type_hrepr__(cls, H, hrepr):
        return H.div['myia-type-bool']('b')


@mixin(Int)
class _Int:
    @classmethod
    def __type_hrepr__(cls, H, hrepr):
        if cls is Int:
            return H.span('Int')
        else:
            return H.span['myia-type-int'](
                H.sub(cls.bits)
            )


@mixin(UInt)
class _UInt:
    @classmethod
    def __type_hrepr__(cls, H, hrepr):
        if cls is UInt:
            return H.span('UInt')
        else:
            return H.span['myia-type-uint'](
                H.sub(cls.bits)
            )


@mixin(Float)
class _Float:
    @classmethod
    def __type_hrepr__(cls, H, hrepr):
        if cls is Float:
            return H.span('Float')
        else:
            return H.span['myia-type-float'](
                H.sub(cls.bits)
            )


################
# Var printers #
################


@mixin(Var)
class _Var:
    def __hrepr__(self, H, hrepr):
        self.ensure_tag()
        return H.div['myia-var'](f'{self.tag}')


@mixin(FilterVar)
class _FilterVar:
    def __hrepr__(self, H, hrepr):
        if hrepr.config.short_vars:
            return Var.__hrepr__(self, H, hrepr)
        else:
            self.ensure_tag()
            return H.div['myia-var', 'myia-filter-var'](
                f'{self.tag}: {self.filter}'
            )


class Subgraph:
    def __init__(self, *nodes, depth=1, context=0):
        nodes = set(nodes)
        for i in range(depth):
            for n in list(nodes):
                nodes.update(n.inputs)
        if context != 0:
            for n in nodes:
                if n.graph is not None:
                    mng = n.graph.manager
            for i in range(context):
                for n in list(nodes):
                    nodes.update(n2 for n2, _ in mng.uses[n])
        self.nodes = nodes

    @classmethod
    def __hrepr_resources__(cls, H):
        """Require the cytoscape plugin for buche."""
        return GraphPrinter.__hrepr_resources__(H)

    def __hrepr__(self, H, hrepr):
        dc = hrepr.config.duplicate_constants
        dfv = hrepr.config.duplicate_free_variables
        fin = hrepr.config.function_in_node
        tgen = hrepr.config.node_tooltip
        cgen = hrepr.config.node_class
        xsty = hrepr.config.graph_style
        gpr = MyiaNodesPrinter(
            self.nodes,
            duplicate_constants=True if dc is None else dc,
            duplicate_free_variables=True if dfv is None else dfv,
            function_in_node=True if fin is None else fin,
            tooltip_gen=tgen,
            class_gen=cgen,
            extra_style=xsty,
        )
        gpr.process()
        return gpr.__hrepr__(H, hrepr)


# @mixin(InferenceTask)
# class _InferenceTask:
#     def __hrepr__(self, H, hrepr):
#         return hrepr.stdrepr_object(
#             'Task',
#             (('wait_for', self._fut_waiter),
#              ('key', self.key)),
#             delimiter="↦",
#         )


@mixin(PendingTentative)
class _PendingTentative:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'PendingTentative',
            (('tentative', self.tentative),
             ('done', self.resolved()),
             ('prio', self.priority()))
        )


#################
# Abstract data #
#################


def _clean(values):
    return {k: v for k, v in values.items()
            if v not in {ANYTHING, UNKNOWN}}


@mixin(AbstractValue)
class _AbstractValue:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            '★Value',
            _clean(self.values).items(),
            delimiter="↦",
            cls='abstract',
        )


@mixin(AbstractScalar)
class _AbstractScalar:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            f'★Scalar',
            _clean(self.values).items(),
            delimiter="↦",
            cls='abstract',
        )


@mixin(AbstractFunction)
class _AbstractFunction:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            self.values[VALUE],
            before='★Function',
            cls='abstract',
        )


@mixin(AbstractJTagged)
class _AbstractJTagged:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            [
                H.div(
                    hrepr.stdrepr_object(
                        '', _clean(self.values).items(), delimiter="↦",
                        cls='noborder'
                    ),
                    hrepr(self.element),
                    style='display:flex;flex-direction:column;'
                )
            ],
            before='★J',
            cls='abstract',
        )


@mixin(AbstractTuple)
class _AbstractTuple:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            self.elements,
            before='★T',
            cls='abstract',
        )


@mixin(AbstractUnion)
class _AbstractUnion:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            self.options,
            before='★U',
            cls='abstract',
        )


@mixin(AbstractArray)
class _AbstractArray:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            [
                H.div(
                    hrepr.stdrepr_object(
                        '', _clean(self.values).items(), delimiter="↦",
                        cls='noborder'
                    ),
                    hrepr(self.element),
                    style='display:flex;flex-direction:column;'
                )
            ],
            before='★A',
            cls='abstract',
        )


@mixin(AbstractList)
class _AbstractList:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            [
                H.div(
                    hrepr.stdrepr_object(
                        '', _clean(self.values).items(), delimiter="↦",
                        cls='noborder'
                    ),
                    hrepr(self.element),
                    style='display:flex;flex-direction:column;'
                )
            ],
            before='★L',
            cls='abstract',
        )


@mixin(AbstractClass)
class _AbstractClass:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            f'★{self.tag}',
            self.attributes.items(),
            delimiter="↦",
            cls='abstract'
        )


@mixin(GraphFunction)
class _GraphFunction:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'GraphFunction',
            (('graph', self.graph), ('context', self.context)),
            delimiter="↦",
        )


@mixin(PartialApplication)
class _PartialApplication:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'Partial',
            (('fn', self.fn), ('args', self.args)),
            delimiter="↦",
        )


@mixin(TypedPrimitive)
class _TypedPrimitive:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'TypedPrimitive',
            (('prim', self.prim),
             ('args', self.args),
             ('output', self.output)),
            delimiter="↦",
        )


@mixin(MetaGraphFunction)
class _MetaGraphFunction:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'MetaGraphFunction',
            (('metagraph', self.metagraph),),
            delimiter="↦",
        )


@mixin(PrimitiveFunction)
class _PrimitiveFunction:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'PrimitiveFunction',
            (('prim', self.prim),),
            delimiter="↦",
        )
