
from typing import NamedTuple
from functools import reduce
from collections import defaultdict
import operator
from .anf_ir import ANFNode, Constant, Graph
from . import primops as P
from .unify import var, Unification, Var
from .debug.utils import mixin
from .graph_utils import dfs
from .anf_ir_utils import \
    succ_deep, exclude_from_set, is_constant, is_constant_graph


_unification = Unification()
unify = _unification.unify


def isvar(x):
    return isinstance(x, Var)


@mixin(Var)
class _Var:
    def __lshift__(self, pattern):
        return SubPattern(self, pattern)


class SubPattern:
    def __init__(self, var, pattern):
        self.var = var
        self.pattern = pattern


def valuevar():
    return var(filter=is_constant)


def fnvar():
    return var(filter=is_constant_graph)


X = var()
Y = var()
Z = var()
X1 = var()
Y1 = var()
Z1 = var()
X2 = var()
Y2 = var()
Z2 = var()
V = valuevar()
V1 = valuevar()
V2 = valuevar()
L = fnvar()


class PatternOpt:
    def __init__(self, pattern, handler):
        self.pattern = pattern
        self.handler = handler

    def _match(self, node, pattern, U):
        if isinstance(node, ANFNode):
            touches = {node}
        else:
            touches = set()
        if isinstance(pattern, SubPattern):
            touches, U = self._match(node, pattern.pattern, U)
            if U:
                U = unify(pattern.var, node, U)
                return touches, U
            return touches, False
        elif isinstance(pattern, tuple):
            sexp = list(node.inputs)
            if sexp:
                if ... in pattern:
                    idx = pattern.index(...)
                    ntail = len(pattern) - idx - 1
                    pattern = pattern[:idx] + pattern[idx + 1:]
                    idx_last = len(sexp) - ntail
                    mid = list(sexp[idx - 1:idx_last])
                    sexp = sexp[:idx - 1] + [mid] + sexp[idx_last:]
                if len(sexp) != len(pattern):
                    return touches, False
                for p, e in zip(pattern, sexp):
                    t2, U = self._match(e, p, U)
                    touches |= t2
                    if U is None:
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
        elif is_constant(node):
            return touches, unify(pattern, node.value, U)
        else:
            return touches, unify(pattern, node, U)

    def match(self, node):
        return self._match(node, self.pattern, {})

    def __call__(self, mutator, univ, node):
        touches, m = self.match(node)
        if m is not False:
            # kwargs = {k.token: v for k, v in m.items()}
            repl = self.handler(node, m)
            if repl is None or repl is node:
                pass
            elif isinstance(repl, ANFNode):
                mutator.replace(node, repl)
                return set()
            else:
                raise TypeError('Wrong return value for pattern.')
        return touches


def pattern_opt(*pattern):
    if len(pattern) == 2 and pattern[0] == 'just':
        pattern = pattern[1]

    def wrap(handler):
        return PatternOpt(pattern, handler)
    return wrap


def simple_sub(pattern, replacement):
    def handler(node, equiv):
        g = node.graph

        def make_node(spec):
            if spec in equiv:
                return equiv[spec]
            elif isvar(spec):
                raise Exception(f'Unresolved variable: {spec}')
            elif spec == ():
                return Constant(())
            elif isinstance(spec, tuple):
                return g.apply(*map(make_node, spec))
            else:
                return Constant(spec)

        return make_node(replacement)

    return PatternOpt(pattern, handler)


class RelinkOperation(NamedTuple):
    node: ANFNode
    key: int
    old: ANFNode
    new: ANFNode

    def inverse(self):
        return Relink(self.node, self.key, self.new, self.old)


class GraphMutator:
    def __init__(self):
        self.listeners = set()

    def commit(self, operations):
        try:
            for i, op in enumerate(operations):
                self._commit_operation(op)
        except Exception as e:
            for op in reversed(operations[:i]):
                self._commit_operation(op.inverse())
            raise
        self.broadcast(operations)

    def commit_operation(self, op):
        self._commit_operation(op)
        self.broadcast([op])

    def _commit_operation(self, op):
        if isinstance(op, RelinkOperation):
            op.node.inputs[op.key] = op.new
        else:
            raise TypeError(f'Expected RelinkOperation.')

    def destroy(self, node):
        node.inputs.clear()

    def broadcast(self, messages):
        for listener in self.listeners:
            listener(messages)

    def relink_operations(self, node, key, new):
        return [RelinkOperation(node, key, node.inputs[key], new)]

    def replace_operations(self, node, new):
        ops = [self.relink_operations(use, key, new)
               for use, key in node.uses]
        return reduce(operator.add, ops, [])

    def relink(self, node, key, new):
        self.commit(self.relink_operations(node, key, new))

    def replace(self, node, new):
        self.commit(self.replace_operations(node, new))

    def transaction(self, *listeners):
        return Transaction(self, listeners)


class Transaction:
    def __init__(self, mutator, listeners=set()):
        self.entered = False
        self.mutator = mutator
        self.listeners = set(listeners) - mutator.listeners
        self.log = []

    def relink(self, node, key, new):
        self.log += self.mutator.relink_operations(node, key, new)

    def replace(self, node, new):
        self.log += self.mutator.replace_operations(node, new)

    def __enter__(self):
        assert not self.entered
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.mutator.listeners |= self.listeners
            try:
                self.mutator.commit(self.log)
            finally:
                self.mutator.listeners -= self.listeners
        else:
            # On exception, throw away all changes
            pass


class EquilibriumTransformer:
    def __init__(self,
                 universe,
                 graphs,
                 transformers):
        self.universe = universe
        self.mutator = GraphMutator()
        self.graphs = set(graphs)
        self.roots = [g.output for g in graphs]
        self.transformers = transformers
        self.repools = defaultdict(set)

    def mark_change(self, node):
        for n in self.repools[node]:
            self.pool.add(n)
        self.repools[node] = set()

    def check_eliminate(self, node):
        if not node.uses:
            inputs = list(node.inputs)
            self.mutator.destroy(node)
            for inp in inputs:
                self.check_eliminate(inp)
            if node in self.watched:
                self.watched.remove(node)

    def _listen_mark_changes(self, ops):
        for op in ops:
            if isinstance(op, RelinkOperation):
                self.mark_change(op.node)
            else:
                raise ValueError(f'Expected RelinkOperation.')

    def _listen_eliminate(self, ops):
        check = {op.old for op in ops}
        for node in check:
            self.check_eliminate(node)

    def _watch(self, node):
        node_list = set(dfs(node, succ_deep, exclude_from_set(self.watched)))
        for node2 in node_list:
            if node2 not in self.watched:
                self.watched.add(node2)
                self.pool.add(node2)
        self.pool.add(node)

    def _listen_new_nodes(self, ops):
        for op in ops:
            if isinstance(op, RelinkOperation):
                self._watch(op.new)
            else:
                raise ValueError(f'Expected RelinkOperation.')

    def process(self, node):
        # Whenever a node changes in the touches set, patterns that
        # failed to run on the current node might now succeed.
        touches = set()
        for transformer in self.transformers:
            # Transformer returns the nodes it has touched, and a
            # list of operations.

            with self.mutator.transaction(self._listen_mark_changes,
                                          self._listen_eliminate,
                                          self._listen_new_nodes) as mut:
                try:
                    ts = transformer(mut, self.universe, node)
                except Exception as e:
                    ts = set()
                    # node.debug.errors.add(e)
                    raise

            if mut.log:
                # Done with this node
                break

            touches |= ts
        else:
            for n in touches:
                self.repools[n].add(node)

    def run(self):
        self.pool = set()
        for root in self.roots:
            self.pool |= set(dfs(root, succ_deep))
        self.watched = set(self.pool)
        while self.pool:
            node = self.pool.pop()
            self.process(node)
