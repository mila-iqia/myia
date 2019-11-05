"""Dead data elimination."""

from collections import defaultdict
from itertools import chain

from .. import abstract, xtype
from ..abstract import ANYTHING, DEAD, PartialApplication
from ..ir import Constant, Graph
from ..operations import Primitive, primitives as P
from ..utils import Named, Partializable, Registry, new_universe, newenv

WILDCARD = Named('WILDCARD')


#####################
# Value propagation #
#####################


class ValuePropagator:

    def __init__(self, resources, root):
        self.resources = resources
        self.manager = resources.manager
        self.need = defaultdict(set)
        self.flow = defaultdict(set)
        self.backflow = defaultdict(set)
        self.results = defaultdict(lambda: defaultdict(set))
        self.todo = set()
        self.run(root)

    def run(self, root):
        for p in root.parameters:
            self.propagate(p, WILDCARD, ANYTHING)
        for ct in self.manager.all_nodes:
            if ct.is_constant():
                if isinstance(ct.value, tuple):
                    self.propagate(ct, WILDCARD, ANYTHING)
                else:
                    self.propagate(ct, ANYTHING, ct.value)
        self.declare_need(root.return_, ANYTHING)
        i = 0
        while self.todo:
            nxt = self.todo.pop()
            self.process_node(nxt)

    def propagate(self, node, need, value):
        if need is ANYTHING and isinstance(value, tuple):
            for i, v in enumerate(value):
                self.propagate(node, (i,), v)
        self._propagate(node, need, value)

    def _propagate(self, node, need, value):
        assert need != ()
        results = self.results[node][need]
        if value not in results:
            results.add(value)
            self.invalidate(node)
            for other_node in self.flow[node]:
                self._propagate(other_node, need, value)

    def invalidate(self, node):
        for node2, i in self.manager.uses[node]:
            self.todo.add(node2)

    def declare_need(self, node, need):
        needs = self.need[node]
        if isinstance(need, tuple) and len(need) > 3:
            need = need[:3]
        if need not in needs:
            if isinstance(need, tuple):
                for i in range(1, len(need)):
                    if need[:i] in needs:
                        return
            needs.add(need)
            self.todo.add(node)
            self.invalidate(node)
            for other_node in self.backflow[node]:
                self.declare_need(other_node, need)

    def process_node(self, node):

        def _dofn(fn, inp):
            if isinstance(fn, Primitive):
                for need in self.need[node]:
                    need_registry[fn](self, need, inp, node)
                    compute_registry[fn](self, need, inp, node)
            elif isinstance(fn, Graph):
                for param, inp in zip(fn.parameters, node.inputs[1:]):
                    self.connect(inp, param)
                self.connect(fn.return_, node)
            elif isinstance(fn, PartialApplication):
                return _dofn(fn.fn, fn.args + inp)
            else:
                raise NotImplementedError(type(fn))

        if node.is_apply():
            f, *inp = node.inputs
            self.declare_need(f, ANYTHING)
            for fn in self.values(f, ANYTHING):
                _dofn(fn, inp)
        else:
            pass

    def connect(self, frm, to):
        if to in self.flow[frm]:
            return
        self.flow[frm].add(to)
        self.backflow[to].add(frm)
        for need in self.need[to]:
            self.declare_need(frm, need)
        for need, values in self.results[frm].items():
            for value in values:
                self._propagate(to, need, value)

    def values(self, node, need):
        return self.results[node][need] | self.results[node][WILDCARD]


#####################
# Need relationship #
#####################


_generic_primitives = (
    P.scalar_eq, P.scalar_ne,
    P.scalar_gt, P.scalar_lt, P.scalar_ge, P.scalar_le,
    P.scalar_sub, P.scalar_mul, P.scalar_div, P.scalar_add, P.scalar_mod,
    P.scalar_pow, P.scalar_exp, P.scalar_log, P.scalar_tanh,
    P.distribute, P.dot, P.transpose, P.scalar_to_array,
    P.bool_and, P.bool_or, P.bool_not, P.bool_eq, P.scalar_usub, P.scalar_uadd,
    P.array_getitem, P.array_setitem, P.scalar_cast, P.reshape, P.scalar_floor,
    P.hastag, P.make_exception,
    P.argmax, P.array_max, P.gather, P.scatter, P.scatter_add,
    P.scalar_sign, P.split, P.concat, P.array_cast,
    P.conv2d, P.conv2d_input_grad, P.conv2d_weight_grad, P.array_to_scalar,
    P.max_pool2d_grad, P.max_pool2d,
    P.handle,  # TODO: special handling
)


need_registry = Registry()
regneed = need_registry.register


compute_registry = Registry()
regcompute = compute_registry.register


@regneed(P.make_tuple)
def _need_make_tuple(engine, need, inputs, out):
    if need is ANYTHING:
        for inp in inputs:
            engine.declare_need(inp, ANYTHING)
    else:
        assert isinstance(need, tuple)
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING
        engine.declare_need(inputs[here], others)


@regcompute(P.make_tuple)
def _compute_make_tuple(engine, need, inputs, out):
    if need is ANYTHING:
        engine.propagate(out, need, tuple(ANYTHING for inp in inputs))
    else:
        assert isinstance(need, tuple)
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING
        for val in engine.values(inputs[here], others):
            engine.propagate(out, need, val)


@regneed(P.tuple_getitem)
def _need_tuple_getitem(engine, need, inputs, out):
    need_tup = need
    if need is ANYTHING:
        need_tup = ()
    tup, idx = inputs
    idxv = engine.values(idx, ANYTHING)
    for v in idxv:
        engine.declare_need(tup, (v, *need_tup))
    engine.declare_need(idx, ANYTHING)


@regcompute(P.tuple_getitem)
def _compute_tuple_getitem(engine, need, inputs, out):
    tup, idx = inputs
    need_tup = need
    if need_tup is ANYTHING:
        need_tup = ()
    for i in engine.values(idx, ANYTHING):
        for v in engine.values(tup, (i, *need_tup)):
            engine.propagate(out, need, v)


@regneed(P.tuple_setitem)
def _need_tuple_setitem(engine, need, inputs, out):
    tup, idx, val = inputs
    engine.declare_need(idx, ANYTHING)

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for v in engine.values(idx, ANYTHING):
        if here is None:
            engine.declare_need(val, others)
            engine.declare_need(tup, need)
        elif v == here:
            engine.declare_need(val, others)
        else:
            engine.declare_need(tup, need)


@regcompute(P.tuple_setitem)
def _compute_tuple_setitem(engine, need, inputs, out):
    tup, idx, val = inputs

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for i in engine.values(idx, ANYTHING):
        if here is None:
            for v in engine.values(val, ANYTHING):
                engine.propagate(out, need, v)
            for v in engine.values(tup, need):
                engine.propagate(out, need, v)
        elif i == here:
            for v in engine.values(val, ANYTHING):
                engine.propagate(out, need, v)
        else:
            for v in engine.values(tup, need):
                engine.propagate(out, need, v)


@regneed(P.env_getitem)
def _need_env_getitem(engine, need, inputs, out):
    env, item, dflt = inputs
    engine.declare_need(dflt, need)
    engine.declare_need(item, ANYTHING)
    if need is ANYTHING:
        need = ()
    for v in engine.values(item, ANYTHING):
        engine.declare_need(env, (v, *need))


@regcompute(P.env_getitem)
def _compute_env_getitem(engine, need, inputs, out):
    # print(need)
    env, item, dflt = inputs
    for v in engine.values(dflt, need):
        engine.propagate(out, need, v)
    need_tup = need
    if need_tup is ANYTHING:
        need_tup = ()
    for i in engine.values(item, ANYTHING):
        for v in engine.values(env, (i, *need_tup)):
            engine.propagate(out, need, v)


@regneed(P.env_setitem)
def _need_env_setitem(engine, need, inputs, out):
    env, item, val = inputs
    engine.declare_need(item, ANYTHING)

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for v in engine.values(item, ANYTHING):
        if here is None:
            engine.declare_need(val, others)
            engine.declare_need(env, need)
        elif v == here:
            # engine.declare_need(val, ANYTHING)
            # engine.declare_need(env, others)
            engine.declare_need(val, others)
        else:
            engine.declare_need(env, need)


@regcompute(P.env_setitem)
def _compute_env_setitem(engine, need, inputs, out):
    env, item, val = inputs

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for i in engine.values(item, ANYTHING):
        if i == here:
            # for v in engine.values(val, ANYTHING):
            for v in engine.values(val, others):
                engine.propagate(out, need, v)
        else:
            for v in engine.values(env, need):
                engine.propagate(out, need, v)


@regneed(P.universe_getitem)
def _need_universe_getitem(engine, need, inputs, out):
    need_tup = need
    if need is ANYTHING:
        need_tup = ()
    u, h = inputs
    hv = engine.values(h, ANYTHING)
    for v in hv:
        engine.declare_need(u, (v, *need_tup))
    engine.declare_need(h, ANYTHING)


@regcompute(P.universe_getitem)
def _compute_universe_getitem(engine, need, inputs, out):
    u, h = inputs
    need_tup = need
    if need_tup is ANYTHING:
        need_tup = ()
    for i in engine.values(h, ANYTHING):
        for v in engine.values(u, (i, *need_tup)):
            engine.propagate(out, need, v)


@regneed(P.universe_setitem)
def _need_universe_setitem(engine, need, inputs, out):
    u, h, val = inputs
    engine.declare_need(h, ANYTHING)

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for v in engine.values(h, ANYTHING):
        assert v is ANYTHING  # TODO: relax this assumption
        if v == here or here is None:
            engine.declare_need(val, others)
        engine.declare_need(u, need)


@regcompute(P.universe_setitem)
def _compute_universe_setitem(engine, need, inputs, out):
    u, h, val = inputs

    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING

    for i in engine.values(h, ANYTHING):
        if i == here:
            for v in engine.values(val, ANYTHING):
                engine.propagate(out, need, v)
        for v in engine.values(u, need):
            engine.propagate(out, need, v)


@regneed(P.partial)
def _need_partial(engine, need, inputs, out):
    fn_node, *args = inputs
    engine.declare_need(fn_node, ANYTHING)
    for fn in engine.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            for p, arg in zip(fn.parameters, args):
                engine.connect(arg, p)
        else:
            raise NotImplementedError(type(fn))


@regcompute(P.partial)
def _compute_partial(engine, need, inputs, out):
    fn_node, *args = inputs
    for fn in engine.values(fn_node, ANYTHING):
        part = PartialApplication(fn, args)
        engine.propagate(out, need, part)


@regneed(P.return_)
def _need_return(engine, need, inputs, out):
    arg, = inputs
    engine.declare_need(arg, need)


@regcompute(P.return_)
def _compute_return_(engine, need, inputs, out):
    arg, = inputs
    for v in engine.values(arg, need):
        engine.propagate(out, need, v)


@regneed(P.raise_)
def _need_raise_(engine, need, inputs, out):
    arg, = inputs
    engine.declare_need(arg, need)


@regcompute(P.raise_)
def _compute_raise_(engine, need, inputs, out):
    pass


@regneed(P.switch)
def _need_switch(engine, need, inputs, out):
    cond, tb, fb = inputs
    engine.declare_need(cond, ANYTHING)
    engine.declare_need(tb, need)
    engine.declare_need(fb, need)


@regcompute(P.switch)
def _compute_switch(engine, need, inputs, out):
    cond, tb, fb = inputs
    for v in chain(engine.values(tb, need), engine.values(fb, need)):
        engine.propagate(out, need, v)


@regneed(P.array_map)
def _need_array_map(engine, need, inputs, out):
    for inp in inputs:
        engine.declare_need(inp, ANYTHING)
    fn_node, *_ = inputs
    for fn in engine.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            engine.declare_need(fn.return_, ANYTHING)
        else:
            raise NotImplementedError(type(fn))


@regcompute(P.array_map)
def _compute_array_map(engine, need, inputs, out):
    engine.propagate(out, need, ANYTHING)


@regneed(P.array_reduce)
def _need_array_reduce(engine, need, inputs, out):
    for inp in inputs:
        engine.declare_need(inp, ANYTHING)
    fn_node, *_ = inputs
    for fn in engine.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            engine.declare_need(fn.return_, ANYTHING)
        else:
            raise NotImplementedError(type(fn))


@regcompute(P.array_reduce)
def _compute_array_reduce(engine, need, inputs, out):
    engine.propagate(out, need, ANYTHING)


@regneed(P.casttag)
def _need_casttag(engine, need, inputs, out):
    arg, tag = inputs
    engine.declare_need(tag, ANYTHING)
    engine.declare_need(arg, need)


@regcompute(P.casttag)
def _compute_casttag(engine, need, inputs, out):
    arg, tag = inputs
    for v in engine.values(arg, need):
        engine.propagate(out, need, v)


@regneed(P.tagged)
def _need_tagged(engine, need, inputs, out):
    arg, tag = inputs
    engine.declare_need(tag, ANYTHING)
    engine.declare_need(arg, need)


@regcompute(P.tagged)
def _compute_tagged(engine, need, inputs, out):
    arg, tag = inputs
    for v in engine.values(arg, need):
        engine.propagate(out, need, v)


@regneed(P.unsafe_static_cast)
def _need_unsafe_static_cast(engine, need, inputs, out):
    arg, typ = inputs
    engine.declare_need(typ, ANYTHING)
    engine.declare_need(arg, need)


@regcompute(P.unsafe_static_cast)
def _compute_unsafe_static_cast(engine, need, inputs, out):
    arg, typ = inputs
    for v in engine.values(arg, need):
        engine.propagate(out, need, v)


@regneed(*_generic_primitives)
def _need_generic(engine, need, inputs, out):
    for inp in inputs:
        engine.declare_need(inp, ANYTHING)


@regcompute(*_generic_primitives)
def _compute_generic(engine, need, inputs, out):
    engine.propagate(out, need, ANYTHING)


#######################
# DeadDataElimination #
#######################


class DeadDataElimination2(Partializable):
    """Eliminate expressions that compute unretrieved data."""

    def __init__(self, resources=None):
        """Initialize a DeadDataElimination."""
        self.resources = resources

    def make_dead(self, node):
        """Create a dead version of the node."""
        a = node.abstract
        val = DEAD
        if isinstance(a, abstract.AbstractScalar):
            if a.xtype() == xtype.EnvType:
                val = newenv
            elif a.xtype() == xtype.UniverseType:
                val = new_universe
        repl = Constant(val)
        repl.abstract = node.abstract
        return repl

    def __call__(self, root):
        """Apply dead data elimination."""
        root.manager.keep_roots(root)
        vprop = ValuePropagator(self.resources, root)
        present = {node for node, needs in vprop.need.items() if needs}
        missing = vprop.manager.all_nodes - present
        mng = root.manager
        for node in missing:
            g = node.graph
            if g not in mng.graphs:
                # This might happen if replace removes a graph.
                continue
            if g and node is g.return_:
                continue
            repl = self.make_dead(node)
            mng.replace(node, repl)
        return False  # Pretend there are no changes, for now


__all__ = [
    'DeadDataElimination2',
]
