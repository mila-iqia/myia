
import networkx as nx
from ..symbols import builtins, inst_builtin
from ..stx import Transformer, \
    Symbol, ValueNode, ApplyNode, LambdaNode, LetNode, \
    ClosureNode, TupleNode, MyiaSyntaxError, TMP, \
    is_global, is_builtin, GenSym, create_lambda
from ..util import Singleton
from types import FunctionType
from ..lib import Record, ZERO
from ..parse import parse_function
from numpy import ndarray
from ..transform import a_normal
from copy import copy
from buche import buche
from unification import var, unify
from collections import defaultdict


ogen = GenSym('global::optimized')


###############
# Edge labels #
###############


# x ==FN=> y      : y is the function used to compute x
# x ==CL=> y      : y is the function for closure x
# x ==IN(i)=> y   : y is the ith argument to the func that computes x
# x ==ELEM(i)=> y : x[i] is y
# x ==GET(i)=> y  : x is y[i]


class FN(Singleton):
    pass


class CL(Singleton):
    pass


class COPY(Singleton):
    pass


FN = FN()
CL = CL()
COPY = COPY()


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


class ELEM:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'.{self.index}='

    def __hash__(self):
        return hash(ELEM) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


class GET:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'.{self.index}'

    def __hash__(self):
        return hash(GET) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


class IRGraph(nx.MultiDiGraph):

    def link(self, dest, source, role):
        assert not isinstance(dest, Symbol)
        assert not isinstance(source, Symbol)
        self.add_edge(dest, source, role)

    def productor(self, node):
        e = self.edges(node, keys=True)
        fn = None
        cl = None
        cp = None
        inputs = {}
        elems = {}
        idx = None
        for n, node2, key in e:
            if key is FN:
                fn = node2
            elif key is CL:
                cl = node2
            elif key is COPY:
                cp = node2
            elif isinstance(key, IN):
                inputs[key.index] = node2
            elif isinstance(key, ELEM):
                elems[key.index] = node2
            elif isinstance(key, GET):
                idx = (node2, key.index)
            else:
                pass
        if len([x for x in [fn, cl, elems, idx, cp] if x]) > 1:
            raise Exception('Same node cannot be produced in multiple ways.')
        if fn:
            return 'apply', fn, tuple(inputs[i] for i in range(len(inputs)))
        elif cl:
            return 'closure', cl, tuple(inputs[i] for i in range(len(inputs)))
        elif elems:
            return 'tuple', tuple(elems[i] for i in range(len(elems)))
        elif idx:
            return tuple(['index', *idx])
        elif cp:
            return tuple(['copy', cp])
        else:
            return None,

    def toposort(self):
        return nx.topological_sort(self)


class IRNode:
    def __init__(self, tag):
        self.tag = tag
        self.values = None

    def to_ast(self):
        return self.tag

    def rename(self, gen):
        if isinstance(self.tag, ValueNode):
            return self
        if isinstance(self.tag, Symbol) and is_global(self.tag):
            return self
        n = copy(self)
        n.tag = gen.dup(self.tag)
        return n

    @property
    def value(self):
        v, = self.values
        return v

    def __str__(self):
        return str(f'{self.__class__.__name__}({self.tag})')

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return hrepr(self.tag)


class IRInput(IRNode):
    pass


class IRComputation(IRNode):
    pass


class IRValue(IRNode):
    def __init__(self, value):
        super().__init__(str(value))
        self.values = [value]

    def to_ast(self):
        v = self.value
        if isinstance(v, Symbol):
            return v
        else:
            return ValueNode(v)


X = var('X')
Y = var('Y')


class PatternOpt:
    def __init__(self, pattern, handler):
        self.pattern = pattern
        self.handler = handler

    def _match(self, lbda, node, pattern, U):
        if isinstance(pattern, tuple):
            mode, *args = lbda.graph.productor(node)
            if mode == 'apply':
                fn, args = args
                sexp = (fn,) + args
                if len(sexp) != len(pattern):
                    return False
                for p, e in zip(pattern, sexp):
                    U = self._match(lbda, e, p, U)
                    if U is False:
                        return False
                return U
            else:
                return False
        elif node.values and len(node.values) == 1:
            return unify(pattern, node.value, U)
        else:
            return unify(pattern, node, U)

    def match(self, lbda, node):
        return self._match(lbda, node, self.pattern, {})

    def __call__(self, lbda, node):
        m = self.match(lbda, node)
        if m:
            kwargs = {k.token: v for k, v in m.items()}
            repl = self.handler(lbda, node, **kwargs)
            if repl is True or repl is False:
                return repl
            else:
                lbda.replace(node, repl)
                return True
        return False


all_patterns = {}


def pattern_opt(*pattern):
    def wrap(handler):
        opt = PatternOpt(pattern, handler)
        all_patterns[handler.__name__] = opt
        return opt
    return wrap


@pattern_opt(builtins.multiply, 1.0, X)
def multiply_by_one(lbda, node, X):
    return X


@pattern_opt(builtins.index, X, 1)
def index_equiv(lbda, node, X):
    lbda.graph.remove_edges_from(list(lbda.graph.edges(node, keys=True)))
    lbda.graph.add_edge(node, X, GET(1))
    return True


@pattern_opt(builtins.Jinv, (builtins.J, X))
def J_cancel(lbda, node, X):
    return X


@pattern_opt(builtins.closure_args, X)
def take_closure_args(lbda, node, X):
    mode, *args = lbda.graph.productor(X)
    if mode is 'closure':
        _, args = args
        lbda.graph.remove_edges_from(list(lbda.graph.edges(node, keys=True)))
        for i, a in enumerate(args):
            lbda.graph.link(node, a, ELEM(i))
        return True
    else:
        return False


class IRLambda(IRNode):
    def __init__(self, universe, tag, gen, inputs, output, lbda):
        super().__init__(tag)
        self.universe = universe
        self.gen = gen
        self.inputs = inputs
        self.output = output
        self.lbda = lbda
        self.graph = IRGraph()
        self.values = [lbda]

    def dup(self):
        g = self.graph
        corresp = {n: n.rename(self.gen) for n in g.nodes}
        for i in self.inputs:
            corresp.setdefault(i, i.rename(self.gen))
        lbda = copy(self)
        g2 = nx.relabel_nodes(g, corresp, copy=True)
        lbda.graph = IRGraph()
        lbda.graph.add_edges_from(g2.edges(keys=True))
        lbda.inputs = [corresp[i] for i in self.inputs]
        lbda.output = corresp[self.output]
        return lbda

    def prune(self):
        to_remove = set(self.graph.nodes)
        to_remove -= set(nx.dfs_preorder_nodes(self.graph, self.output))
        if to_remove:
            self.graph.remove_nodes_from(to_remove)
            return True
        return False

    def merge(self, n1, n2):
        if n1 is n2:
            return True
        if type(n1) != type(n2):
            return False
        if isinstance(n1, IRLambda):
            return False
        if self._is_constant(n1):
            if n1.value == n2.value:
                self.replace(n1, n2)
                return True
            return False
        else:
            edges1 = {k: n for _, n, k in self.graph.edges(n1, keys=True)}
            edges2 = {k: n for _, n, k in self.graph.edges(n2, keys=True)}
            if edges1.keys() != edges2.keys():
                return False
            for k, n3 in edges1.items():
                if not self.merge(n3, edges2[k]):
                    return False
            self.replace(n1, n2)
            return True

    def cse(self):
        node_to_hash = {}
        hash_to_node = defaultdict(list)
        for n in reversed(list(nx.topological_sort(self.graph))):
            if isinstance(n, (IRLambda, IRInput)):
                h = hash(n)
            elif self._is_constant(n):
                h = hash(n.value) ^ hash(type(n.value))
            else:
                h = 0
                for _, n2, k in self.graph.edges(n, keys=True):
                    h ^= node_to_hash[n2] ^ hash(k)
            node_to_hash[n] = h
            hash_to_node[h].append(n)
        for h, (node0, *nodes) in hash_to_node.items():
            for n in nodes:
                self.merge(n, node0)

    def _inline_data(self, node):
        g = self.graph
        mode, *args = g.productor(node)
        if mode == 'apply':
            fn, args = args
            if isinstance(fn, IRLambda):
                return fn, args
            else:
                mode2, *args2 = g.productor(fn)
                if mode2 == 'closure':
                    fn2, args2 = args2
                    if isinstance(fn2, IRLambda):
                        return fn2, args2 + args
        return None

    def inlinable(self, node):
        return self._inline_data(node) is not None

    def inline(self, node):
        g = self.graph
        _d = self._inline_data(node)
        if _d is None:
            return False
        f, args = _d
        f2 = f.dup()

        g.add_edges_from(f2.graph.edges(keys=True))

        for input, arg in zip(f2.inputs, args):
            g.link(input, arg, COPY)

        g.remove_edges_from(list(g.edges(node, keys=True)))
        g.link(node, f2.output, COPY)

        return True

    def inline_all(self):
        changes = False
        for node in list(self.graph.nodes):
            if self.inline(node):
                changes = True
        return changes

    def replace(self, old, new):
        g = self.graph
        for n, _, k in list(g.in_edges(old, keys=True)):
            g.remove_edge(n, old, k)
            g.add_edge(n, new, k)
        if old is self.output:
            self.output = new

    def _is_constant(self, node):
        return isinstance(node, IRLambda) or \
            (node.values and len(node.values) == 1)

    def value_to_node(self, v):
        from ..interpret import Function
        from ..lib import Closure

        if isinstance(v, Function):
            an = a_normal(v.ast)
            an.primal = v.ast.primal
            return self.universe.acquire(an)
        if isinstance(v, Closure):
            cl = self.value_to_node(v.fn)
            n = IRComputation(ogen('<clos>'))
            self.graph.link(n, cl, CL)
            for i, arg in enumerate(v.args):
                a = self.value_to_node(arg)
                self.graph.link(n, a, IN(i))
            return n
        elif isinstance(v, (int, float, str)):
            return IRValue(v)
        else:
            raise Exception(f'Do not recognize value: {v}')

    def evaluate_constants(self):
        changes = False
        for n in list(self.graph.nodes):
            mode, *args = self.graph.productor(n)
            if mode == 'apply':
                f, args = args
                all_constant = all(self._is_constant(x) for x in (f,) + args)
                if all_constant:
                    eenv = self.universe.eenv
                    if isinstance(f, IRLambda):
                        fn = eenv.evaluate(f.lbda)
                    else:
                        fn = eenv.evaluate(f.value)
                    res = fn(*[eenv.import_value(arg.value)
                               for arg in args])
                    n2 = self.value_to_node(res)
                    self.replace(n, n2)
        return changes

    def collapse_copies(self):
        changes = False
        for a, b, k in list(self.graph.edges(keys=True)):
            if k is COPY:
                changes = True
                self.replace(a, b)
        return changes

    def collapse_index(self):
        changes = False
        for a, b, k in list(self.graph.edges(keys=True)):
            if isinstance(k, GET):
                for _, d, k2 in list(self.graph.edges(b, keys=True)):
                    if isinstance(k2, ELEM) and k.index == k2.index:
                        changes = True
                        self.replace(a, d)
        return changes

    def simplify(self):
        changes = False
        for n in list(self.graph.nodes):
            for opt in all_patterns.values():
                changes |= opt(self, n)
        return changes


class GlobalResolver(dict):
    # TODO: merge this with EvaluationEnv

    def __init__(self, object_map, pool):
        self.object_map = object_map
        self.pool = pool
        self.accepted_types = (bool, int, float,
                               ndarray, list, tuple, Record, str)

    def import_value(self, v):
        try:
            x = self.object_map[v]
        except (TypeError, KeyError):
            pass
        else:
            return self.import_value(x)

        if isinstance(v, Symbol) and is_builtin(v):
            return v

        # try:
        #     x = self.primitives[v]
        # except (TypeError, KeyError):
        #     pass
        # else:
        #     return x

        if isinstance(v, self.accepted_types) or v is ZERO or v is None:
            return v
        elif isinstance(v, (type, FunctionType)):
            # Note: Python's FunctionType, i.e. actual Python functions
            try:
                lbda = parse_function(v)
            except (TypeError, OSError):
                raise ValueError(f'Myia cannot interpret value: {v}')
            return self.import_value(lbda)
        elif isinstance(v, LambdaNode):
            return v
        elif isinstance(v, Symbol):
            raise ValueError(f'Myia cannot resolve {v} '
                             f'from namespace {v.namespace}')
        else:
            raise ValueError(f'Myia cannot interpret value: {v}')

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            if isinstance(item, Symbol) and is_builtin(item):
                self[item] = item
            else:
                self[item] = self.import_value(self.pool[item])
            return self[item]


class Universe:
    def __init__(self, resolver, eenv):
        self.resolver = resolver
        self.eenv = eenv
        self.anf_lbdas = {}
        self.dependencies = nx.DiGraph()

    def acquire(self, anf_lbda):
        if anf_lbda in self.anf_lbdas:
            return self.anf_lbdas[anf_lbda]
        irlbda = self._graph_transform(anf_lbda)
        # # NOTE: _graph_transform sets the key in self.anf_lbdas
        # self.anf_lbdas[anf_lbda] = irlbda
        return irlbda

    def aggressively_optimize(self, anf_lbda):
        lbda = self.anf_lbdas[anf_lbda]
        changes = True
        while changes:
            # Run until equilibrium (all the optimizations below
            # are normalizing)
            opers = [lbda.evaluate_constants(),
                     lbda.inline_all(),
                     lbda.collapse_copies(),
                     lbda.collapse_index(),
                     lbda.simplify(),
                     lbda.prune(),
                     lbda.cse()]
            changes = any(opers)

    def _graph_transform(self, lbda):
        let = lbda.body
        assert isinstance(let, LetNode)

        assoc = {}

        def wrap(x, cls=IRComputation):
            if not isinstance(x, (Symbol, ValueNode)):
                raise Exception('Should be a Symbol or ValueNode.', x)
            values = None

            if x in assoc:
                return assoc[x]

            if isinstance(x, ValueNode):
                values = {x.value}
            elif is_global(x):
                if is_builtin(x):
                    values = {x}
                else:
                    imp = self.resolver[x]
                    if isinstance(imp, LambdaNode):
                        imp2 = a_normal(imp)
                        imp2.primal = imp.primal
                        irlbda = self.acquire(imp2)
                        assoc[x] = irlbda
                        return irlbda
                    elif isinstance(imp, Symbol) and is_builtin(imp):
                        values = {imp}
                    else:
                        raise Exception('Invalid thing to import.')

            if isinstance(x, Symbol):
                n = assoc.setdefault(x, cls(x))
            else:
                n = cls(x)
            n.values = values
            return n

        inputs = [wrap(sym, IRInput) for sym in lbda.args]
        output = wrap(let.body, IRComputation)
        irl = IRLambda(self, ogen(lbda.ref, '!'), lbda.gen,
                       inputs, output, lbda)

        self.anf_lbdas[lbda] = irl
        self.dependencies.add_node(irl)

        g = irl.graph

        def assign(k, v, rel=None):
            if isinstance(k, TupleNode):
                tmp = lbda.gen(TMP)
                assign(tmp, v, rel)
                for i, kk in enumerate(k.values):
                    assign(kk, tmp, GET(i))
                return

            wk = wrap(k)
            if rel is not None:
                g.link(wk, wrap(v), rel)
                return

            if isinstance(v, ApplyNode):
                g.link(wk, wrap(v.fn), FN)
                for i, arg in enumerate(v.args):
                    g.link(wk, wrap(arg), IN(i))
            elif isinstance(v, (Symbol, ValueNode)):
                g.link(wk, wrap(v), COPY)
            elif isinstance(v, ClosureNode):
                g.link(wk, wrap(v.fn), CL)
                for i, arg in enumerate(v.args):
                    g.link(wk, wrap(arg), IN(i))
            elif isinstance(v, TupleNode):
                for i, value in enumerate(v.values):
                    g.link(wk, wrap(value), ELEM(i))
            else:
                raise MyiaSyntaxError('Illegal ANF clause.', node=v)

        for k, v in let.bindings:
            assign(k, v)

        return irl

    @property
    def lbdas(self):
        return list(self.dependencies.nodes)

    def export(self, anf_lbda):
        irlbda = self.anf_lbdas[anf_lbda]
        return GraphToANF().transform(irlbda)

    def export_all(self):
        return {lbda: self.export(lbda) for lbda in self.anf_lbdas.keys()}


class GraphToANF:
    def __init__(self):
        pass

    def transform(self, irl):
        g = irl.graph
        nodes = g.toposort()
        bindings = []

        for node in nodes:
            assert not isinstance(node, Symbol)

            if isinstance(node, IRLambda):
                pass

            else:
                mode, *args = g.productor(node)
                if mode == 'apply':
                    f, args = args
                    args = [a.to_ast() for a in args]
                    val = ApplyNode(f.to_ast(), *args)
                elif mode == 'closure':
                    f, args = args
                    args = [a.to_ast() for a in args]
                    val = ClosureNode(f.to_ast(), args)
                elif mode == 'tuple':
                    args, = args
                    args = [a.to_ast() for a in args]
                    val = TupleNode(args)
                elif mode == 'index':
                    arg, idx = args
                    val = ApplyNode(inst_builtin.index,
                                    arg.to_ast(),
                                    ValueNode(idx))
                elif mode == 'copy':
                    arg, = args
                    val = arg.to_ast()
                else:
                    continue
                bindings.append((node.tag, val))
        bindings.reverse()

        l = create_lambda(irl.tag,
                          [i.tag for i in irl.inputs],
                          LetNode(bindings, irl.output.tag),
                          irl.gen)
        l.primal = irl.lbda.primal
        return l
