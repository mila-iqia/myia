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

    def propagate_all(self, node, need, values):
        for value in values:
            self.propagate(node, need, value)

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
                    vprop_registry[fn](self, need, inp, node)
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


def _split_need(need):
    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING
    return here, others


def _need_tup(need):
    if need is ANYTHING:
        return ()
    return need


vprop_registry = Registry()
regvprop = vprop_registry.register


@regvprop(P.make_tuple)
def _vprop_make_tuple(engine, need, inputs, out):
    # Need
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

    # Compute
    if need is ANYTHING:
        engine.propagate(out, need, tuple(ANYTHING for inp in inputs))
    else:
        assert isinstance(need, tuple)
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING
        engine.propagate_all(out, need, engine.values(inputs[here], others))


@regvprop(P.tuple_getitem)
def _vprop_tuple_getitem(engine, need, inputs, out):
    coll, key = inputs
    need_tup = _need_tup(need)
    keyv = engine.values(key, ANYTHING)

    # Need
    for i in keyv:
        engine.declare_need(coll, (i, *need_tup))
    engine.declare_need(key, ANYTHING)

    # Compute
    for i in keyv:
        engine.propagate_all(out, need, engine.values(coll, (i, *need_tup)))


@regvprop(P.tuple_setitem)
def _vprop_tuple_setitem(engine, need, inputs, out):
    coll, key, val = inputs
    here, others = _split_need(need)
    if here is None:
        propval, propcoll = True, True
    else:
        matches = {i == here for i in engine.values(key, ANYTHING)}
        propval = True in matches
        propcoll = False in matches

    # Need
    engine.declare_need(key, ANYTHING)
    if propval:
        engine.declare_need(val, others)
    if propcoll:
        engine.declare_need(coll, need)

    # Compute
    if propval:
        engine.propagate_all(out, need, engine.values(val, ANYTHING))
    if propcoll:
        engine.propagate_all(out, need, engine.values(coll, need))


@regvprop(P.env_getitem)
def _vprop_env_getitem(engine, need, inputs, out):
    coll, key, dflt = inputs
    need_tup = _need_tup(need)
    keyv = engine.values(key, ANYTHING)

    # Need
    engine.declare_need(dflt, need)
    engine.declare_need(key, ANYTHING)
    for i in keyv:
        engine.declare_need(coll, (i, *need_tup))

    # Compute
    for v in engine.values(dflt, need):
        engine.propagate(out, need, v)
    for i in keyv:
        engine.propagate_all(out, need, engine.values(coll, (i, *need_tup)))


@regvprop(P.env_setitem)
def _vprop_env_setitem(engine, need, inputs, out):
    coll, key, val = inputs
    here, others = _split_need(need)
    if here is None:
        propval, propcoll = True, True
    else:
        matches = {i == here for i in engine.values(key, ANYTHING)}
        propval = True in matches
        propcoll = False in matches

    # Need
    engine.declare_need(key, ANYTHING)
    if propval:
        engine.declare_need(val, others)
    if propcoll:
        engine.declare_need(coll, need)

    # Compute
    if propval:
        engine.propagate_all(out, need, engine.values(val, ANYTHING))
    if propcoll:
        engine.propagate_all(out, need, engine.values(coll, need))


@regvprop(P.universe_getitem)
def _vprop_universe_getitem(engine, need, inputs, out):
    u, h = inputs
    need_tup = _need_tup(need)

    # Need
    hv = engine.values(h, ANYTHING)
    for v in hv:
        engine.declare_need(u, (v, *need_tup))
    engine.declare_need(h, ANYTHING)

    # Compute
    for i in engine.values(h, ANYTHING):
        engine.propagate_all(out, need, engine.values(u, (i, *need_tup)))


@regvprop(P.universe_setitem)
def _vprop_universe_setitem(engine, need, inputs, out):
    coll, key, val = inputs
    here, others = _split_need(need)
    propcoll = True
    if here is None:
        propval = True
    else:
        assert here is ANYTHING
        matches = {i == here for i in engine.values(key, ANYTHING)}
        propval = True in matches

    # Need
    engine.declare_need(key, ANYTHING)
    if propval:
        engine.declare_need(val, others)
    if propcoll:
        engine.declare_need(coll, need)

    # Compute
    if propval:
        engine.propagate_all(out, need, engine.values(val, ANYTHING))
    if propcoll:
        engine.propagate_all(out, need, engine.values(coll, need))


@regvprop(P.partial)
def _vprop_partial(engine, need, inputs, out):
    fn_node, *args = inputs

    # Need
    engine.declare_need(fn_node, ANYTHING)
    for fn in engine.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            for p, arg in zip(fn.parameters, args):
                engine.connect(arg, p)
        else:
            raise NotImplementedError(type(fn))

    # Compute
    for fn in engine.values(fn_node, ANYTHING):
        part = PartialApplication(fn, args)
        engine.propagate(out, need, part)


@regvprop(P.return_)
def _vprop_return(engine, need, inputs, out):
    arg, = inputs

    # Need
    engine.declare_need(arg, need)

    # Compute
    engine.propagate_all(out, need, engine.values(arg, need))


@regvprop(P.raise_)
def _vprop_raise_(engine, need, inputs, out):
    arg, = inputs

    # Need
    engine.declare_need(arg, ANYTHING)

    # No compute


@regvprop(P.switch)
def _vprop_switch(engine, need, inputs, out):
    cond, tb, fb = inputs

    # Need
    engine.declare_need(cond, ANYTHING)
    engine.declare_need(tb, need)
    engine.declare_need(fb, need)

    # Compute
    engine.propagate_all(out, need, engine.values(tb, need))
    engine.propagate_all(out, need, engine.values(fb, need))


@regvprop(P.array_map)
def _vprop_array_map(engine, need, inputs, out):
    # Need
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

    # Compute
    engine.propagate(out, need, ANYTHING)


@regvprop(P.array_reduce)
def _vprop_array_reduce(engine, need, inputs, out):
    # Need
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

    # Compute
    engine.propagate(out, need, ANYTHING)


@regvprop(P.casttag, P.tagged, P.unsafe_static_cast)
def _vprop_cast_operation(engine, need, inputs, out):
    arg, tag = inputs

    # Need
    engine.declare_need(tag, ANYTHING)
    engine.declare_need(arg, need)

    # Compute
    engine.propagate_all(out, need, engine.values(arg, need))


@regvprop(
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
def _vprop_generic(engine, need, inputs, out):
    # Need
    for inp in inputs:
        engine.declare_need(inp, ANYTHING)

    # Compute
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
