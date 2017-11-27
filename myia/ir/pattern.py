
from ..inference.types import var, unify, isvar
from ..symbols import builtins
from .graph import IRNode, IRGraph, NO_VALUE
from collections import defaultdict


def valuevar(name):
    # return var(name, lambda x: x.value is not NO_VALUE)
    return var(name, lambda x: x.is_constant())


def fnvar(name):
    # return var(name, lambda x: x.value and isinstance(x.value, IRGraph))
    return var(name, lambda x: x.is_graph())


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


pattern_bank = {}


def pattern_opt(*pattern):
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


class EquilibriumPass:
    def __init__(self, *patterns):
        self.patterns = patterns

    def __call__(self, universe, graph):
        eq = EquilibriumTransformer(universe, [graph], self.patterns)
        eq.run()
