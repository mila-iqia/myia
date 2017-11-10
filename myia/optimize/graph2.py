
from buche import buche
from collections import defaultdict
from ..transform import a_normal
from ..util import Singleton
from ..interpret import EvaluationEnv
from ..stx import top as about_top, Transformer, \
    Symbol, ValueNode, ApplyNode, LambdaNode, LetNode, \
    ClosureNode, TupleNode, MyiaSyntaxError, TMP, ANORM, \
    is_global, is_builtin, GenSym, create_lambda
from ..lib import Record, ZERO, Primitive
from ..symbols import builtins
from numpy import ndarray
import json
import os


NO_VALUE = None


ogen = GenSym('global::optimized')
ggen = GenSym('global::graph')


_css_path = f'{os.path.dirname(__file__)}/graph.css'
_css = open(_css_path).read()


# Roles
# FN: unique, function used to compute node
# IN(idx): unique, idxth input to computation for the node


class FN(Singleton):
    pass


class IN:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'IN({self.index})'

    def __hash__(self):
        return hash(IN) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


FN = FN()    # type: ignore


class IRNode:
    def __init__(self, graph, tag):
        # Graph the node belongs to
        self.graph = graph
        # Node name
        self.tag = tag
        # Outgoing edges
        self.fn = None
        self.inputs = []
        # Incoming edges as (role, node) tuples
        self.users = set()
        # Value this node will take, if it can be determined
        self.value = NO_VALUE
        # Information inferred about this node (type, shape, etc.)
        self.inferred = {}
        # Optimization/source code trace
        self.about = about_top()

    def succ(self, role):
        if role is FN:
            return {self.fn}
        elif isinstance(role, IN):
            return {self.inputs[role.index]}
        else:
            raise KeyError(f'Invalid role: {role}')

    def set_succ(self, role, node, old_node=None):
        assert isinstance(node, IRNode) or node is None
        if role is FN:
            old_node = self.fn
            self.fn = node
        elif isinstance(role, IN):
            old_node = self.inputs[role.index]
            self.inputs[role.index] = node
        else:
            raise KeyError(f'Invalid role: {role}')
        if old_node is not None:
            old_node.users.remove((role, self))
        if node is not None:
            node.users.add((role, self))

    def set_operation(self, fn, inputs=None):
        self.set_succ(FN, fn)
        for i, inp in enumerate(self.inputs):
            if inp is not None:
                inp.users.remove((IN(i), self))
        if fn is None:
            self.inputs = []
        else:
            self.inputs = list(inputs)
            for i, inp in enumerate(self.inputs):
                inp.users.add((IN(i), self))

    def subsume(self, node):
        self.users |= node.users
        for role, n in set(node.users):
            n.set_succ(role, self, node)
        node.users = set()

    def redirect(self, new_node):
        new_node.subsume(self)

    def __getitem__(self, role):
        if role is FN:
            return self.fn
        elif isinstance(role, IN):
            return self.inputs[role.index]
        else:
            raise KeyError(f'Invalid role: {role}')

    def __setitem__(self, role, node):
        self.set_succ(role, node)

    def __hrepr__(self, H, hrepr):
        if self.value is NO_VALUE:
            return hrepr(self.tag)
        else:
            return hrepr(self.value)


class IRGraph:
    def __init__(self, parent, tag, inputs, output, gen):
        self.parent = parent
        self.tag = tag
        self.inputs = tuple(inputs)
        self.output = output
        self.gen = gen

    def contained_in(self, parent):
        g = self
        while g:
            if g is parent:
                return True
            g = g.parent
        return False

    def link(self, node1, node2, role):
        for node in [node1, node2]:
            if not isinstance(node, IRNode):
                raise TypeError(f'link(...) must be called on IRNode'
                                ' instances.')
            # if node.graph is not self:
            #     raise ValueError(f'link(...) must be called on nodes that'
            #                      ' belong to the graph.')
        node1.succ(role, node2)

    def replace(self, node1, node2):
        node1.redirect(node2)
        if node1 is self.output:
            self.output = node2

    def iternodes(self, boundary=False):
        # Basic BFS from output node
        to_visit = {self.output}
        seen = set()
        while to_visit:
            node = to_visit.pop()
            if not node or node in seen:
                continue
            if node.graph is not self:
                if boundary and node.graph:
                    yield node
                else:
                    continue
            yield node
            seen.add(node)
            to_visit.add(node.fn)
            for inp in node.inputs:
                to_visit.add(inp)

    def iterboundary(self):
        return self.iternodes(self, True)

    def __hrepr__(self, H, hrepr):
        rval = H.graphElement(H.style(_css))
        options = {
            'layout': {
                'name': 'dagre',
                'rankDir': 'TB'
            }
        }
        rval = rval(H.options(json.dumps(options)))

        opts = {
            'duplicate_constants': hrepr.config.duplicate_constants,
            'function_in_node': hrepr.config.function_in_node,
            'follow_references': hrepr.config.follow_references
        }
        nodes_data, edges_data = GraphPrinter({self}, **opts).process()
        for elem in nodes_data + edges_data:
            rval = rval(H.element(json.dumps(elem)))
        return rval


class GraphPrinter:
    def __init__(self,
                 entry_points,
                 duplicate_constants=True,
                 function_in_node=True,
                 follow_references=True):
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        self.pool = set()
        self.currid = 0
        self.ids = {}
        self.nodes = []
        self.edges = []

    def next_id(self):
        self.currid += 1
        return f'X{self.currid}'

    def register(self, obj):
        if not self.should_dup(obj) and obj in self.ids:
            return False, self.ids[obj]
        id = self.next_id()
        self.ids[obj] = xzz = id
        return True, id

    def const_fn(self, node):
        cond = self.function_in_node \
            and node.fn \
            and node.fn.value is not NO_VALUE
        if cond:
            return node.fn.tag

    def should_dup(self, node):
        return isinstance(node, IRNode) \
            and self.duplicate_constants \
            and node.value is not NO_VALUE

    def add_graph(self, g):
        new, id = self.register(g)
        if new:
            self.nodes.append({'data': {'id': id, 'label': str(g.tag)},
                               'classes': 'function'})
        return id

    def add_node(self, node, g=None):

        new, id = self.register(node)
        if not new:
            return id

        if not g:
            g = node.graph

        if node.value is NO_VALUE:
            lbl = str(node.tag)
        elif isinstance(node.value, IRGraph):
            if self.follow_references:
                self.graphs.add(node.value)
            lbl = str(node.tag)
        else:
            lbl = str(node.value)

        if node.graph is None:
            cl = 'constant'
        elif node is g.output:
            cl = 'output'
        elif node in g.inputs:
            cl = 'input'
        else:
            cl = 'intermediate'

        cfn = self.const_fn(node)
        if cfn:
            if '/out' in lbl or '/in' in lbl:
                lbl = ""
            lbl = f'{lbl}:{cfn}'

        data = {'id': id, 'label': lbl}
        if g:
            data['parent'] = self.add_graph(g)
        self.nodes.append({'data': data, 'classes': cl})
        self.pool.add(node)
        return id

    def process_graph(self, g):
        self.add_node(g.output)

        while self.pool:
            node = self.pool.pop()
            if self.const_fn(node):
                if self.follow_references \
                        and isinstance(node.fn.value, IRGraph):
                    self.graphs.add(node.fn.value)
                edges = []
            else:
                edges = [(node, FN, node.fn)] \
                    if node.fn is not NO_VALUE \
                    else []
            edges += [(node, IN(i), inp)
                      for i, inp in enumerate(node.inputs) or []]
            for edge in edges:
                src, role, dest = edge
                if role is FN:
                    lbl = 'F'
                else:
                    lbl = str(role.index)
                dest_id = self.add_node(dest, self.should_dup(dest) and g)
                data = {
                    'id': self.next_id(),
                    'label': lbl,
                    'source': dest_id,
                    'target': self.ids[src]
                }
                self.edges.append({'data': data})

    def process(self):
        while self.graphs:
            g = self.graphs.pop()
            self.process_graph(g)
        return self.nodes, self.edges


class IREvaluationEnv(EvaluationEnv):

    def wrap_local(self, sym, g):
        node = self[sym]
        node.graph = g
        return node

    def convert_value(self, v):
        accepted_types = (bool, int, float,
                          ndarray, list, tuple, Record, str)
        if isinstance(v, accepted_types) or v is ZERO or v is None:
            n = IRNode(None, ggen('value'))
            n.value = v
            return n
        elif isinstance(v, Symbol) and is_builtin(v):
            n = IRNode(None, v)
            n.value = v
            return n
        elif isinstance(v, Symbol):
            return IRNode(None, v)
            # raise ValueError(f'Myia cannot resolve {v} '
            #                  f'from namespace {v.namespace}')
        elif isinstance(v, ValueNode):
            return self.convert_value(v.value)
        else:
            raise TypeError(f'Myia cannot convert value: {v}')

    def compile(self, _lbda):
        lbda = a_normal(_lbda)

        let = lbda.body
        assert isinstance(let, LetNode)

        ref = lbda.ref
        if ref.relation is ANORM:
            ref = ref.label

        g = IRGraph(None, ref,
                    [], None, lbda.gen)
        g.inputs = tuple(self.wrap_local(sym, g) for sym in lbda.args)
        g.output = self.wrap_local(let.body, g)
        rval = IRNode(None, g.tag)
        self.compile_cache[_lbda] = rval

        def assign(k, v, idx=None):
            if isinstance(k, TupleNode):
                tmp = lbda.gen(TMP)
                assign(tmp, v, idx)
                for i, kk in enumerate(k.values):
                    assign(kk, tmp, i)
                return

            wk = self.wrap_local(k, g)
            if idx is not None:
                wk.set_operation(self[builtins.index], [self[v], self[idx]])
                return

            if isinstance(v, ApplyNode):
                args = [self[a] for a in v.args]
                wk.set_operation(self[v.fn], args)
            elif isinstance(v, (Symbol, ValueNode)):
                wk.set_operation(self[builtins.identity], [self[v]])
            elif isinstance(v, ClosureNode):
                args = [self[v.fn]] + [self[a] for a in v.args]
                wk.set_operation(self[builtins.partial], args)
            elif isinstance(v, TupleNode):
                args = [self[a] for a in v.values]
                wk.set_operation(self[builtins.mktuple], args)
            else:
                raise MyiaSyntaxError('Illegal ANF clause.', node=v)

        for k, v in let.bindings:
            assign(k, v)

        rval.value = g
        return rval


class EquilibriumTransformer:
    def __init__(self,
                 roots,
                 taggers,
                 transformers,
                 follow=lambda _: True):
        self.roots = roots
        self.tags = {}

    def run(self):
        pass


class Universe:
    def __init__(self, eenv):
        self.eenv = eenv
        self.irenv = IREvaluationEnv({}, eenv.pool)
        self.entry_points = set()
        self.anf_lbdas = {}

    def acquire(self, anf_lbda):
        if anf_lbda in self.anf_lbdas:
            return self.anf_lbdas[anf_lbda]
        irlbda = self._graph_transform(anf_lbda)
        # # NOTE: _graph_transform sets the key in self.anf_lbdas
        # self.anf_lbdas[anf_lbda] = irlbda
        self.entry_points.add(irlbda)
        return irlbda

    def _graph_transform(self, lbda):
        return self.irenv.compile(lbda).value

    def closure_unconvert(self):
        pool = set(self.entry_points)
        while pool:
            g = pool.pop()
            for node in g.iternodes():
                if node.fn and node.fn.value is builtins.partial:
                    fnn = node.inputs[0]
                    fn = fnn.value
                    if isinstance(fn, IRGraph):
                        g2 = fn
                        g.replace(node, node.inputs[0])
                        for i, g2i in zip(node.inputs[1:], g2.inputs):
                            g2.replace(g2i, i)
                        g2.inputs = g2.inputs[len(node.inputs) - 1:]
                        pool.add(g2)
                    elif fn in self.eenv.primitives:
                        sym = fn
                        fn = self.eenv.primitives[fn]
                        if isinstance(fn, Primitive):
                            g2 = IRGraph(None, ogen(sym, '~'),
                                         [], None, g.gen)
                            clins = node.inputs[1:]
                            argins = [IRNode(g2, ogen(argn))
                                      for argn in fn.argnames[len(clins):]]
                            g2.inputs = clins + argins
                            o = IRNode(g2, ogen('/out'))
                            g2.output = o
                            o.set_operation(fnn, g2.inputs)
                            g2n = IRNode(None, g2.tag)
                            g2n.value = g2
                            g.replace(node, g2n)
                elif isinstance(node.value, IRGraph):
                    pool.add(node.value)
                elif node.fn and isinstance(node.fn.value, IRGraph):
                    pool.add(node.fn.value)

    def equilibrium_transform(self, transformers, follow_nodes):
        pass

    # def _closure_convert(self, g):
    #     closure_args = []
    #     while True:
    #         changes = False
    #         for node in g.iterboundary():
    #             if isinstance(node.value, IRGraph):
    #                 clos = self._closure_convert(node.value)
    #                 opers.append((node, clos))

    #                 new = IRNode(g, ogen('CLOS'))
    #                 g.replace(node, new)
    #                 new.set_operation(partial, [node] + clos)

    #         if not changes:
    #             break

    #     opers = []
    #     targets = []

    #     for node in g.iterboundary():
    #         if node.graph:
    #             closure_args.append(node)
    #             new_arg = IRNode(g, ogen(node.tag))
    #             g.replace(node, new_arg)
    #         elif isinstance(node.value, IRGraph):
    #             clos = self._closure_convert(node.value)
    #             new = IRNode(g, ogen('CLOS'))
    #             g.replace(node, new)
    #             new.set_operation(partial, [node] + clos)
    #     g.inputs = closure_args + g.inputs
    #     return closure_args

    # def closure_convert(self):
    #     pool = set(self.entry_points)
    #     while pool:
    #         g = pool.pop()
    #         for node in g.iterboundary():
