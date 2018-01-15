
from ..anf_ir import Graph, Apply, Constant
import os


css_path = f'{os.path.dirname(__file__)}/graph.css'
css = open(css_path).read()


def is_computation(x):
    return isinstance(x, Apply)


def is_constant(x):
    return isinstance(x, Constant)


def is_graph(x):
    return isinstance(x, Constant) and isinstance(x.value, Graph)


class GraphPrinter:
    def __init__(self,
                 entry_points,
                 duplicate_constants=False,
                 function_in_node=False,
                 follow_references=False):
        # Graphs left to process
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        # Nodes processed
        self.processed = set()
        # Nodes left to process
        self.pool = set()
        # Nodes and edges are accumulated in these lists
        self.nodes = []
        self.edges = []
        # Nodes that are to be colored as return nodes
        self.returns = set()
        # IDs for duplicated constants
        self.currid = 0
        # Custom rules for nodes that represent certain calls
        self.custom_rules = {
            'return': self.process_node_return
        }

    def id(self, x):
        return f'X{id(x)}'

    def fresh_id(self):
        self.currid += 1
        return f'Y{self.currid}'

    def cynode(self, id, label, classes, parent=None):
        if not isinstance(id, str):
            id = self.id(id)
        data = {'id': id, 'label': str(label)}
        if parent:
            parent = parent if isinstance(parent, str) else self.id(parent)
            data['parent'] = parent
        self.nodes.append({'data': data, 'classes': classes})

    def cyedge(self, src_id, dest_id, label):
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
        self.edges.append({'data': data})

    def const_fn(self, node):
        fn = node.inputs[0] if node.inputs else None
        if fn and is_constant(fn):
            if is_graph(fn):
                return fn.value.debug.debug_name
            else:
                return str(fn.value)
        else:
            return None

    def add_graph(self, g):
        if g in self.processed:
            return
        name = g.debug.debug_name
        argnames = [p.debug.debug_name for p in g.parameters]
        lbl = f'{name}({", ".join(argnames)})'
        self.cynode(id=g, label=lbl, classes='function')
        self.processed.add(g)

    def label(self, node):
        if is_graph(node):
            lbl = node.value.debug.debug_name
        elif is_constant(node):
            lbl = str(node.value)
        else:
            lbl = ''
            if self.function_in_node:
                cfn = self.const_fn(node)
                if cfn:
                    lbl = cfn

            if node.debug.name:
                if lbl:
                    lbl += f'→{node.debug.name}'
                else:
                    lbl = node.debug.name
        return lbl or '·'

    def process_node_return(self, node, g, cl):
        self.cynode(id=node, label='', parent=g, classes='const_output')
        ret = node.inputs[1]
        self.process_edges([(node, '', ret)])

    def process_node_generic(self, node, g, cl):
        lbl = self.label(node)

        self.cynode(id=node, label=lbl, parent=g, classes=cl)

        fn = node.inputs[0] if node.inputs else None
        if fn and is_graph(fn):
            self.graphs.add(fn.value)

        edges = []
        if fn and not (is_constant(fn) and self.function_in_node):
            edges.append((node, 'F', fn))

        edges += [(node, str(i + 1), inp)
                  for i, inp in enumerate(node.inputs[1:]) or []]

        self.process_edges(edges)

    def process_node(self, node):
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
            else:
                self.pool.add(dest)
                self.cyedge(src_id=src, dest_id=dest, label=lbl)

    def process_graph(self, g):
        self.add_graph(g)
        for inp in g.parameters:
            self.process_node(inp)

        ret = g.return_.inputs[1]
        if not is_computation(ret):
            ret = g.return_

        self.returns.add(ret)
        self.pool.add(ret)

        while self.pool:
            node = self.pool.pop()
            self.process_node(node)

    def process(self):
        if self.nodes or self.edges:
            return
        while self.graphs:
            g = self.graphs.pop()
            self.process_graph(g)
        return self.nodes, self.edges

    def follow(self, node):
        if is_graph(node) and self.follow_references:
            self.graphs.add(node.value)
