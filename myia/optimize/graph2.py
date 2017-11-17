
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
from ..inference.types import var, unify, isvar
import json
import os


class NO_VALUE(Singleton):
    pass


NO_VALUE = NO_VALUE()   # type: ignore


ogen = GenSym('global::optimized')
ggen = GenSym('global::graph')


_css_path = f'{os.path.dirname(__file__)}/graph.css'
_css = open(_css_path).read()


# Roles
# FN: unique, function used to compute node
# IN(idx): unique, idxth input to computation for the node


class FN(Singleton):
    """
    Edge label for the FN relation between two nodes.
    """
    pass


class IN:
    """
    Edge label for the IN(i) relation between two nodes.
    """
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
    """
    Node in the intermediate representation. It can represent:

    * A computation:
      - fn != None, inputs == list of inputs, value == NO_VALUE
    * A numeric/string/etc. constant
      - fn == None, value == the constant
    * A builtin function like add or multiply
      - fn == None, value == the Symbol for the builtin
    * A pointer to another Myia function
      - fn == None, value == an IRGraph
    * An input
      - fn == None, value == NO_VALUE

    Attributes:
        graph: Parent IRGraph for this node. This should be None for
            constant nodes.
        tag: Symbol representing the node name.
        fn: Points to an IRNode providing the function to call.
        inputs: List of IRNode arguments.
        users: Set of incoming edges. Each edge is a (role, node)
            tuple where role is FN or IN(i)
        value: Value taken by this node.
        inferred: TODO
        about: Tracks source code location and sequence of
            transformations and optimizations for this node.
    """
    def __init__(self, graph, tag, value=NO_VALUE):
        # Graph the node belongs to
        self.graph = graph
        # Node name (Symbol instance)
        self.tag = tag
        # Outgoing edges
        self.fn = None
        self.inputs = []
        # Incoming edges as (role, node) tuples
        self.users = set()
        # Value this node will take, if it can be determined
        self.value = value
        # Information inferred about this node (type, shape, etc.)
        self.inferred = {}
        # Optimization/source code trace
        self.about = about_top()

    def successors(self):
        """
        List of nodes that this node depends on.
        """
        succ = [self.fn] + self.inputs
        return [s for s in succ if s]

    def app(self):
        """
        If this node is an application, return:

            (n.fn, *n.inputs)

        Otherwise, return None.
        """
        if not self.fn:
            return None
        else:
            return (self.fn,) + tuple(self.inputs)

    def succ(self, role):
        """
        If role is FN, return node.fn, if role is IN(i), return
        node.inputs[i].
        """
        if role is FN:
            return {self.fn}
        elif isinstance(role, IN):
            return {self.inputs[role.index]}
        else:
            raise KeyError(f'Invalid role: {role}')

    def set_succ(self, role, node):
        """
        Create an edge toward node with given role (FN or IN(i))
        """
        return self._commit(self.set_succ_operations, (role, node))

    def set_app(self, fn, inputs):
        """
        Make this node an application of fn on the specified inputs.
        fn and the inputs must be IRNode instances.
        """
        return self._commit(self.set_app_operations, (fn, inputs))

    def redirect(self, new_node):
        """
        Transfer every user of this node to new_node.
        """
        return self._commit(self.redirect_operations, (new_node,))

    def subsume(self, node):
        """
        Transfer every user of the given node to this one.
        """
        return node.redirect(self)

    # The following methods return a list of atomic "operations" to
    # execute in order to perform the task. Atomic operations are
    # ('link', from, to, role) and ('unlink', from, to, node)

    def set_succ_operations(self, role, node):
        assert isinstance(node, IRNode) or node is None
        if role is FN:
            unl = self.fn
        elif isinstance(role, IN):
            unl = self.inputs[role.index]
        else:
            raise KeyError(f'Invalid role: {role}')
        rval = []
        if unl:
            if unl == node:
                # Nothing to do because the new successor is the
                # same as the old one.
                return []
            else:
                rval.append(('unlink', self, unl, role))
        if node is not None:
            rval.append(('link', self, node, role))
        return rval

    def set_app_operations(self, fn, inputs):
        rval = self.set_succ_operations(FN, fn)
        if fn:
            for i, inp in enumerate(self.inputs):
                if inp is not None:
                    rval.append(('unlink', self, inp, IN(i)))
            for i, inp in enumerate(inputs):
                if inp is not None:
                    rval.append(('link', self, inp, IN(i)))
        return rval

    def redirect_operations(self, node):
        rval = []
        for role, n in set(self.users):
            rval += n.set_succ_operations(role, node)
        rval.append(('redirect', self, node, None))
        return rval

    def process_operation(self, op, node, role):
        # Execute a 'link' or 'unlink' operation.
        if op == 'link':
            if role is FN:
                assert self.fn is None
                self.fn = node
            elif isinstance(role, IN):
                idx = role.index
                nin = len(self.inputs)
                if nin <= idx:
                    self.inputs += [None for _ in range(idx - nin + 1)]
                assert self.inputs[idx] is None
                self.inputs[idx] = node
            node.users.add((role, self))
        elif op == 'unlink':
            if role is FN:
                assert self.fn is node
                self.fn = None
            elif isinstance(role, IN):
                idx = role.index
                nin = len(self.inputs)
                assert self.inputs[idx] is node
                self.inputs[idx] = None
            node.users.remove((role, self))
        elif op == 'redirect':
            pass
        else:
            raise ValueError('Operation must be link or unlink.')

    def _commit(self, fn, args):
        for op, n1, n2, r in fn(*args):
            n1.process_operation(op, n2, r)

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
    """
    Graph with inputs and an output. Represents a Myia function or
    a closure.

    Attributes:
        parent: The IRGraph for the parent function, if this graph
            represents a closure. Otherwise, None.
        tag: A Symbol representing the name of this graph.
        gen: A GenSym instance to generate new tags within this
            graph.
        inputs: A tuple of input IRNodes for this graph.
        output: The IRNode representing the output of this graph.
    """
    def __init__(self, parent, tag, gen):
        self.parent = parent
        self.tag = tag
        self.inputs = []
        self.output = None
        self.gen = gen

    def dup(self, g=None):
        """
        Duplicate this graph, optionally setting g as the parent of
        every node in the graph.

        Return the new graph (or g), a list of inputs, and the output
        node.
        """
        set_io = g is None
        if not g:
            g = IRGraph(self.parent, self.tag, self.gen)
        mapping = {}
        for node in self.inputs + tuple(self.iternodes()):
            mapping[node] = IRNode(g, g.gen(node.tag, '+'), node.value)
        for n1, n2 in mapping.items():
            sexp = n1.app()
            if sexp:
                f, *args = sexp
                f2 = mapping.get(f, f)
                args2 = [mapping.get(a, a) for a in args]
                n2.set_app(f2, args2)
        output = mapping[self.output]
        inputs = [mapping[i] for i in self.inputs]
        if set_io:
            g.output = output
            g.inputs = inputs
        return g, inputs, output

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

    @classmethod
    def __hrepr_resources__(cls, H):
        return H.bucheRequire(name='cytoscape')

    def __hrepr__(self, H, hrepr):
        rval = H.cytoscapeGraph(H.style(_css))
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
    """
    Helper class to print Myia graphs.

    Arguments:
        entry_points: A collection of graphs to print.
        duplicate_constants: If True, each use of a constant will
            be shown as a different node.
        function_in_node: If True, applications of a known function
            will display the function's name in the node like this:
            "node_name:function_name". If False, the function will
            be a separate constant node, with a "F" edge pointing to
            it.
        follow_references: If True, graphs encountered while walking
            the initial graphs will also be processed.
    """
    def __init__(self,
                 entry_points,
                 duplicate_constants=True,
                 function_in_node=True,
                 follow_references=True):
        # Graphs left to process
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        # Nodes left to process
        self.pool = set()
        # ID system for the nodes that will be sent to buche
        self.currid = 0
        self.ids = {}
        # Nodes and edges are accumulated in these lists
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
                    if node.fn is not None \
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
            return IRNode(None, ggen('value'), v)
        elif isinstance(v, Symbol) and is_builtin(v):
            return IRNode(None, v, v)
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

        g = IRGraph(None, ref, lbda.gen)
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
                wk.set_app(self[builtins.index], [self[v], self[idx]])
                return

            if isinstance(v, ApplyNode):
                args = [self[a] for a in v.args]
                wk.set_app(self[v.fn], args)
            elif isinstance(v, (Symbol, ValueNode)):
                wk.set_app(self[builtins.identity], [self[v]])
            elif isinstance(v, ClosureNode):
                args = [self[v.fn]] + [self[a] for a in v.args]
                wk.set_app(self[builtins.partial], args)
            elif isinstance(v, TupleNode):
                args = [self[a] for a in v.values]
                wk.set_app(self[builtins.mktuple], args)
            else:
                raise MyiaSyntaxError('Illegal ANF clause.', node=v)

        for k, v in let.bindings:
            assign(k, v)

        rval.value = g
        return rval


def valuevar(name):
    return var(name, lambda x: x.value is not NO_VALUE)


def fnvar(name):
    return var(name, lambda x: x.value and isinstance(x.value, IRGraph))


X = var('X')
Y = var('Y')
Z = var('Z')
V = valuevar('V')
V1 = valuevar('V1')
V2 = valuevar('V2')
L = fnvar('L')


class PatternOpt:
    def __init__(self, pattern, handler):
        self.pattern = pattern
        self.handler = handler

    def _match(self, node, pattern, U):
        if isinstance(node, IRNode):
            touches = {node}
        else:
            touches = set()
        if isinstance(pattern, tuple):
            sexp = node.app()
            if sexp:
                if ... in pattern:
                    idx = pattern.index(...)
                    ntail = len(pattern) - idx - 1
                    pattern = pattern[:idx] + pattern[idx + 1:]
                    idx_last = len(sexp) - ntail
                    mid = list(sexp[idx - 1:idx_last])
                    sexp = sexp[:idx - 1] + (mid,) + sexp[idx_last:]
                if len(sexp) != len(pattern):
                    return touches, False
                for p, e in zip(pattern, sexp):
                    t2, U = self._match(e, p, U)
                    touches |= t2
                    if U is False:
                        return touches, False
                return touches, U
            else:
                return touches, False
        elif isinstance(node, list):
            if isvar(pattern):
                for x in node:
                    if not unify(pattern, x, U):
                        return touches, False
                return touches, {**U, var(pattern.token): node}
            else:
                return touches, False
        elif isvar(pattern):
            return touches, unify(pattern, node, U)
        elif node.value is not NO_VALUE:
            return touches, unify(pattern, node.value, U)
        else:
            return touches, unify(pattern, node, U)

    def match(self, node):
        return self._match(node, self.pattern, {})

    def __call__(self, univ, node):
        touches, m = self.match(node)
        if m:
            kwargs = {k.token: v for k, v in m.items()}
            repl = self.handler(univ, node, **kwargs)
            if isinstance(repl, list):
                return touches, repl
            elif isinstance(repl, tuple):
                return touches, node.set_app_operations(repl[0], repl[1:])
            elif isinstance(repl, IRNode):
                return touches, node.redirect_operations(repl)
        return touches, []


all_patterns = {}


def pattern_opt(*pattern):
    def wrap(handler):
        opt = PatternOpt(pattern, handler)
        all_patterns[handler.__name__] = opt
        return opt
    return wrap


@pattern_opt(builtins.multiply, 1.0, X)
def multiply_by_one_l(univ, node, X):
    return X


@pattern_opt(builtins.multiply, X, 1.0)
def multiply_by_one_r(univ, node, X):
    return X


@pattern_opt(builtins.identity, X)
def drop_copy(univ, node, X):
    return X


from ..interpret import evaluate


@pattern_opt(V1, V2, ...)
def eval_constant(univ, node, V1, V2):
    if V1.value == builtins.partial:
        return False
    f = V1
    args = V2
    fn = evaluate(f.value)
    res = fn(*[arg.value for arg in args])
    n = IRNode(None, None)
    n.value = res
    return n


class EquilibriumTransformer:
    def __init__(self,
                 universe,
                 graphs,
                 transformers,
                 follow=lambda a, b: True):
        self.universe = universe
        self.graphs = graphs
        self.roots = [g.output for g in graphs]
        self.transformers = transformers
        self.follow = follow
        self.repools = defaultdict(set)

    def mark_change(self, node):
        assert node
        for n in self.repools[node]:
            self.processed.discard(n)
            self.pool.add(n)
        self.repools[node] = set()

    def process(self, node):
        touches = set()
        assert node
        for transformer in self.transformers:
            ts, changes = transformer(self.universe, node)
            touches |= ts
            if changes:
                for op, node1, node2, role in changes:
                    self.mark_change(node1)
                    # self.mark_change(node2)
                    node1.process_operation(op, node2, role)
                    if op is 'redirect':
                        for g in self.graphs:
                            if g.output is node1:
                                g.output = node2
                                self.pool.add(node2)
                break
        else:
            self.processed.add(node)
            for succ in node.successors():
                assert succ
                if succ not in self.processed and self.follow(node, succ):
                    self.pool.add(succ)
            for n in touches:
                self.repools[n].add(node)

    def run(self):
        self.pool = set(self.roots)
        self.processed = set()
        while self.pool:
            node = self.pool.pop()
            self.process(node)


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
                            g2 = IRGraph(None, ogen(sym, '~'), g.gen)
                            clins = node.inputs[1:]
                            argins = [IRNode(g2, ogen(argn))
                                      for argn in fn.argnames[len(clins):]]
                            g2.inputs = clins + argins
                            o = IRNode(g2, ogen('/out'))
                            g2.output = o
                            o.set_app(fnn, g2.inputs)
                            g2n = IRNode(None, g2.tag)
                            g2n.value = g2
                            g.replace(node, g2n)
                elif isinstance(node.value, IRGraph):
                    pool.add(node.value)
                elif node.fn and isinstance(node.fn.value, IRGraph):
                    pool.add(node.fn.value)

    def equilibrium_transform(self):
        transformers = list(all_patterns.values())
        eqt = EquilibriumTransformer(self, self.entry_points, transformers)
        eqt.run()

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
    #                 new.set_app(partial, [node] + clos)

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
    #             new.set_app(partial, [node] + clos)
    #     g.inputs = closure_args + g.inputs
    #     return closure_args

    # def closure_convert(self):
    #     pool = set(self.entry_points)
    #     while pool:
    #         g = pool.pop()
    #         for node in g.iterboundary():
