
from ..lib import Closure
from ..stx import is_global, is_builtin, GenSym
from ..inference.types import var, unify, isvar
from ..symbols import builtins
from .graph import IRNode, IRGraph, NO_VALUE
from collections import defaultdict
from buche import buche


ogen = GenSym('opt::pattern')


def valuevar(name):
    # return var(name, lambda x: x.value is not NO_VALUE)
    return var(name, lambda x: x.is_constant())


def fnvar(name):
    # return var(name, lambda x: x.value and isinstance(x.value, IRGraph))
    return var(name, lambda x: x.is_graph())


def globalvar(name):
    return var(name, lambda x: x.is_constant() and
               is_global(x.value) and
               not is_builtin(x.value))


X = var('X')
Y = var('Y')
Z = var('Z')
V = valuevar('V')
V1 = valuevar('V1')
V2 = valuevar('V2')
GV = globalvar('GV')
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


pattern_bank = {}


def pattern_opt(*pattern):
    if len(pattern) == 2 and pattern[0] == 'just':
        pattern = pattern[1]

    def wrap(handler):
        opt = PatternOpt(pattern, handler)
        pattern_bank[handler.__name__] = opt
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


@pattern_opt(V1, V2, ...)
def eval_constant(univ, node, V1, V2):
    def acq(value):
        if hasattr(value, '__myia_graph__'):
            value = value.__myia_graph__
            tag = value.lbda.ref
            return IRNode(None, tag, value)
        elif isinstance(value, Closure):
            ptial = IRNode(None, builtins.partial, builtins.partial)
            n = IRNode(None, node.tag)
            n.set_app(ptial,
                      [acq(value.fn)] + [acq(arg) for arg in value.args])
            return n
        elif isinstance(value, IRGraph):
            return IRNode(None, value.lbda.ref, value)
        else:
            return IRNode(None, ogen(node.tag, '@'), value)

    univ = univ.universes['const_prop']
    if V1.value == builtins.partial:
        return False
    f = V1
    args = V2
    fn = univ[f.value]
    res = fn(*[univ[arg.value] for arg in args])
    tag = None
    return acq(res)


@pattern_opt('just', GV)
def resolve_global(univ, node, GV):
    univ = univ.universes['opt']
    n = IRNode(None, GV.value, univ[GV.value])
    return n


@pattern_opt(L, X, ...)
def inline(univ, node, L, X):
    g = L.value
    args = X

    g2, inputs, output = g.dup(node.graph)

    idn = IRNode(None, builtins.identity, builtins.identity)
    chgs = []
    for arg, inp in zip(args, inputs):
        chgs += inp.set_app_operations(idn, [arg])
    chgs += node.redirect_operations(output)
    return chgs


@pattern_opt((builtins.partial, X, Y, ...), Z, ...)
def expand_partial_app(univ, node, X, Y, Z):
    f = X
    args = Y + Z
    node2 = IRNode(node.graph, node.tag, node.value)
    ops = node2.set_app_operations(f, args)
    ops += node.redirect_operations(node2)
    return ops


@pattern_opt(builtins.index, (builtins.mktuple, X, ...), V)
def index_into_tuple(univ, node, X, V):
    return X[int(V.value)]


# TODO: (a, b, c) + (d, e, f) => (a+d, b+e, c+f)
# TODO: cancel J/Jinv
# TODO: J(switch)?


class EquilibriumTransformer:
    def __init__(self,
                 universe,
                 graphs,
                 transformers,
                 follow=lambda a, b: True,
                 follow_references=True):
        self.universe = universe
        self.graphs = set(graphs)
        self.roots = [g.output for g in graphs]
        self.transformers = transformers
        self.follow = follow
        self.repools = defaultdict(set)
        self.follow_references = follow_references

    def mark_change(self, node):
        assert node
        for n in self.repools[node]:
            self.processed.discard(n)
            self.pool.add(n)
        self.repools[node] = set()

    def check_eliminate(self, node):
        if len(node.users) > 0:
            return
        if any(g.output is node for g in self.graphs):
            return

        succ = node.successors(True)
        for r, s in succ:
            s.users.remove((r, node))
        for _, s in succ:
            self.check_eliminate(s)

    def process(self, node):
        assert node
        # Whenever a node changes in the touches set, patterns that
        # failed to run on the current node might now succeed.
        touches = set()
        if node.is_graph() and self.follow_references:
            # Run until equilibrium on graphs this graph uses.
            graph = node.value
            self.graphs.add(graph)
            self.pool.add(graph.output)
            return
        for transformer in self.transformers:
            # Transformer returns the nodes it has touched, and a
            # list of operations.
            ts, changes = transformer(self.universe, node)
            if changes:
                for op, node1, node2, role in changes:
                    # We notify that a change happened to the first node.
                    # This will re-trigger any pattern that looked at this
                    # node but failed to change anything.
                    self.mark_change(node1)
                    node1.process_operation(op, node2, role)
                    if op is 'redirect':
                        # We need to adjust graph outputs on redirections.
                        # This should probably be considered a design flaw
                        # in IRNode.
                        for g in self.graphs:
                            if g.output is node1:
                                g.output = node2
                                self.pool.add(node2)
                # Check if this node should be eliminated
                self.check_eliminate(node)
                # Done with this node
                break
            touches |= ts
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


class EquilibriumPass:
    def __init__(self, *patterns):
        self.patterns = patterns

    def __call__(self, universe, graph):
        eq = EquilibriumTransformer(universe, [graph], self.patterns)
        eq.run()
