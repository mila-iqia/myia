"""Dead data elimination."""

from collections import defaultdict

from .. import abstract, xtype
from ..abstract import ANYTHING, DEAD, PartialApplication
from ..ir import Constant, Graph
from ..operations import Primitive, primitives as P
from ..utils import Named, Partializable, Registry, newenv, tracer

WILDCARD = Named("WILDCARD")
MAX_NEED_DEPTH = 5


#####################
# Value propagation #
#####################


class ValuePropagator:
    r"""Perform value propagation.

    Each need for each node is associated to a set of possible values. A "need"
    is either ANYTHING or a path. For example, the need (0, 7) on a node means
    that we need element 7 of element 0 of the tuple returned at this node.

    * Primitives of the form \*_getitem or \*_setitem should
      manipulate the needs accordingly, e.g. if we need (0, 7) on
      getitem(tup, 3), then we need (3, 0, 7) on tup.

    We start with the ANYTHING need on root.return\_, ANYTHING on
    root.parameters, and the appropriate values on constants, and propagate
    until equilibrium. The set of possible needs and values is appropriately
    limited to ensure that the process terminates.

    Some nodes may end up not being needed at all, e.g. if the only need on
    a tuple is (1,), then the node for element 0, if it is used nowhere else,
    is not needed, and it can be removed later.

    """

    def __init__(self, resources, root):
        """Initialize a ValuePropagator and run the algorithm."""
        self.resources = resources
        self.manager = resources.opt_manager
        # need :: node -> {paths}
        self.need = defaultdict(set)
        # flow :: node -> {node}
        self.flow = defaultdict(set)
        # backflow :: node -> {node} (opposite of flow)
        self.backflow = defaultdict(set)
        # results :: node -> need -> values
        self.results = defaultdict(lambda: defaultdict(set))
        self.todo = set()
        self.run(root)

    def run(self, root):
        """Run the algorithm."""
        for p in root.parameters:
            self.add_value(p, WILDCARD, ANYTHING)
        for ct in self.manager.all_nodes:
            if ct.is_constant():
                if isinstance(ct.value, tuple):
                    self.add_value(ct, WILDCARD, ANYTHING)
                else:
                    self.add_value(ct, ANYTHING, ct.value)
        self.add_need(root.return_, ANYTHING)
        while self.todo:
            nxt = self.todo.pop()
            self.process_node(nxt)

    def values(self, node, need):
        """Return the current set of possible values for node:need."""
        # The purpose of the WILDCARD key is to act as a fallback to cover any
        # need. For example, if a parameter is a tree (so its type at this
        # point is a recursive AbstractTuple), we might need paths (1, 0), (1,
        # 1, 0), and so on, but rather than having an infinite number of
        # entries, we just have a single WILDCARD entry that covers them all.
        # This is conceptually different from ANYTHING, which covers the path
        # (), where the need is for the unindexed data structure. WILDCARD
        # is currently only set in run(), for parameters and tuple constants.
        # So most of the time there's nothing in it.
        return self.results[node][need] | self.results[node][WILDCARD]

    def add_value(self, node, need, value):
        """Propagate a value for a need on a node."""
        if need is ANYTHING and isinstance(value, tuple):
            for i, v in enumerate(value):
                self._add_value(node, (i,), v)
        self._add_value(node, need, value)

    def _add_value(self, node, need, value):
        assert need != ()
        results = self.results[node][need]
        if value not in results:
            results.add(value)
            # Invalidate all uses of this node, because they now have new
            # values to test.
            for node2, i in self.manager.uses[node]:
                self.todo.add(node2)
            for other_node in self.flow[node]:
                self._add_value(other_node, need, value)

    def add_need(self, node, need):
        """Declare that node has the given need."""
        needs = self.need[node]
        if isinstance(need, tuple) and len(need) > MAX_NEED_DEPTH:
            need = need[:MAX_NEED_DEPTH]
        if need not in needs:
            if isinstance(need, tuple):
                for i in range(1, len(need)):
                    if need[:i] in needs:
                        return
            needs.add(need)
            self.todo.add(node)
            for other_node in self.backflow[node]:
                self.add_need(other_node, need)

    def process_node(self, node):
        """Perform value/need propagation on node.

        If new values or needs are discovered, the nodes touched by the change
        will be rescheduled for processing. It is important for the process to
        be monotonic and for the sets of values/needs to be finite for it to
        terminate.
        """

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
            self.add_need(f, ANYTHING)
            for fn in self.values(f, ANYTHING):
                _dofn(fn, inp)
        else:
            pass

    def connect(self, frm, to):
        """Connect node frm to node to.

        When calling `y = f(x)`, x is connected to the first parameter of f,
        and the return node of f is connected to y.

        * Any value propagated to frm is propagated to to.
        * Any need on to is propagated to frm.
        """
        if to in self.flow[frm]:
            return
        self.flow[frm].add(to)
        self.backflow[to].add(frm)
        for need in self.need[to]:
            self.add_need(frm, need)
        for need, values in self.results[frm].items():
            for value in values:
                self._add_value(to, need, value)

    def passthrough(self, arg, out, need, *, through_need=None):
        """Declare that out:need is equivalent to arg:through_need.

        * We need through_need on arg.
        * Values for through_need on arg are propagated to need on out.

        Arguments:
            arg: An argument that is related to out.
            out: The output node.
            need: The need on the output node.
            through_need: The equivalent need on arg. If None, then the
                need on out is used.

        """
        if through_need is None:
            through_need = need
        self.add_need(arg, through_need)
        for value in self.values(arg, through_need):
            self.add_value(out, need, value)

    def getitem(self, coll, key, out, need):
        r"""Implement a standard getitem operation.

        * We need key:ANYTHING
        * For all possible values of key, we need coll:(key, \*need),
          and the values are propagated to out:need.

        Arguments:
            coll: The collection node.
            key: The key/index node.
            out: The output node.
            need: The need on out.

        """
        need_tup = need
        if need_tup is ANYTHING:
            need_tup = ()
        self.add_need(key, ANYTHING)
        for i in self.values(key, ANYTHING):
            self.passthrough(coll, out, need, through_need=(i, *need_tup))

    def setitem(self, coll, key, val, out, need, *, precise_values=True):
        """Implement a standard setitem operation.

        * We need key:ANYTHING
        * If the first item of the need matches at least one of the possible
          values for the key, we propagate the rest of the need to val.
        * Otherwise, or if precise_values is False, we propagate the need
          to coll.

        Arguments:
            coll: The collection node.
            key: The key/index node.
            val: The value node, to put in the collection with the given key.
            out: The output node.
            need: The need on out.
            precise_values: Whether the key represents a single possible key
                at runtime, or a set of keys. If it is the former, then if the
                need is equal to the key, nothing is needed from coll. If it is
                the latter, then we declare a need on both val and coll.

        """
        here, others = _split_need(need)
        if here is None:
            propval, propcoll = True, True
        else:
            matches = {i == here for i in self.values(key, ANYTHING)}
            propval = True in matches
            propcoll = not precise_values or False in matches

        self.add_need(key, ANYTHING)
        if propval:
            self.passthrough(val, out, need, through_need=others)
        if propcoll:
            self.passthrough(coll, out, need)


def _split_need(need):
    if need is ANYTHING:
        here, others = None, need
    else:
        here, *others = need
        others = tuple(others)
        if not others:
            others = ANYTHING
    return here, others


vprop_registry = Registry()
regvprop = vprop_registry.register


@regvprop(P.make_tuple)
def _vprop_make_tuple(vprop, need, inputs, out):
    here, others = _split_need(need)
    if here is None:
        for inp in inputs:
            vprop.add_need(inp, ANYTHING)
        vprop.add_value(out, need, tuple(ANYTHING for inp in inputs))
    else:
        vprop.passthrough(inputs[here], out, need, through_need=others)


@regvprop(P.tuple_getitem)
def _vprop_tuple_getitem(vprop, need, inputs, out):
    coll, key = inputs
    vprop.getitem(coll, key, out, need)


@regvprop(P.tuple_setitem)
def _vprop_tuple_setitem(vprop, need, inputs, out):
    coll, key, val = inputs
    vprop.setitem(coll, key, val, out, need)


@regvprop(P.env_getitem)
def _vprop_env_getitem(vprop, need, inputs, out):
    coll, key, dflt = inputs
    vprop.passthrough(dflt, out, need)
    vprop.getitem(coll, key, out, need)


@regvprop(P.env_setitem)
def _vprop_env_setitem(vprop, need, inputs, out):
    coll, key, val = inputs
    vprop.setitem(coll, key, val, out, need)


@regvprop(P.universe_getitem)
def _vprop_universe_getitem(vprop, need, inputs, out):
    coll, key = inputs
    vprop.getitem(coll, key, out, need)


@regvprop(P.universe_setitem)
def _vprop_universe_setitem(vprop, need, inputs, out):
    coll, key, val = inputs
    vprop.setitem(coll, key, val, out, need, precise_values=False)


@regvprop(P.partial)
def _vprop_partial(vprop, need, inputs, out):
    fn_node, *args = inputs
    vprop.add_need(fn_node, ANYTHING)
    for fn in vprop.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            for p, arg in zip(fn.parameters, args):
                vprop.connect(arg, p)
        else:
            raise NotImplementedError(type(fn))

        part = PartialApplication(fn, args)
        vprop.add_value(out, need, part)


@regvprop(P.return_, P.identity)
def _vprop_return(vprop, need, inputs, out):
    (arg,) = inputs
    vprop.passthrough(arg, out, need)


@regvprop(P.raise_)
def _vprop_raise_(vprop, need, inputs, out):
    (arg,) = inputs
    vprop.add_need(arg, ANYTHING)


@regvprop(P.switch)
def _vprop_switch(vprop, need, inputs, out):
    cond, tb, fb = inputs
    vprop.add_need(cond, ANYTHING)
    vprop.passthrough(tb, out, need)
    vprop.passthrough(fb, out, need)


@regvprop(P.array_map, P.array_reduce)
def _vprop_array_operation(vprop, need, inputs, out):
    for inp in inputs:
        vprop.add_need(inp, ANYTHING)
    fn_node, *_ = inputs
    for fn in vprop.values(fn_node, ANYTHING):
        if isinstance(fn, Primitive):
            pass
        elif isinstance(fn, Graph):
            vprop.add_need(fn.return_, ANYTHING)
        else:
            raise NotImplementedError(type(fn))

    vprop.add_value(out, need, ANYTHING)


@regvprop(P.casttag, P.tagged, P.unsafe_static_cast)
def _vprop_cast_operation(vprop, need, inputs, out):
    arg, tag = inputs
    vprop.add_need(tag, ANYTHING)
    vprop.passthrough(arg, out, need)


@regvprop(
    P.argmax,
    P.array_cast,
    P.array_getitem,
    P.array_max,
    P.array_scan,
    P.array_setitem,
    P.array_to_scalar,
    P.bool_and,
    P.bool_eq,
    P.bool_not,
    P.bool_or,
    P.broadcast_shape,
    P.concat,
    P.conv2d,
    P.conv2d_weight_grad,
    P.conv_transpose2d,
    P.stop_gradient,
    P.random_initialize,
    P.random_uint32,
    P.distribute,
    P.dot,
    P.take,
    P.take_grad_inp,
    P.env_add,
    P.gather,
    P.handle,
    P.hastag,
    P.invert_permutation,
    P.make_exception,
    P.max_pool2d,
    P.max_pool2d_grad,
    P.reshape,
    P.scalar_add,
    P.scalar_abs,
    P.scalar_bit_and,
    P.scalar_bit_lshift,
    P.scalar_bit_or,
    P.scalar_bit_rshift,
    P.scalar_bit_xor,
    P.scalar_bit_not,
    P.scalar_cast,
    P.scalar_cos,
    P.scalar_div,
    P.scalar_eq,
    P.scalar_exp,
    P.scalar_floor,
    P.scalar_ge,
    P.scalar_gt,
    P.scalar_le,
    P.scalar_log,
    P.scalar_lt,
    P.scalar_max,
    P.scalar_mod,
    P.scalar_mul,
    P.scalar_ne,
    P.scalar_pow,
    P.scalar_sign,
    P.scalar_sin,
    P.scalar_sub,
    P.scalar_tan,
    P.scalar_tanh,
    P.scalar_to_array,
    P.scalar_trunc,
    P.scalar_uadd,
    P.scalar_usub,
    P.scatter,
    P.scatter_add,
    P.shape,
    P.split,
    P.transpose,
)
def _vprop_generic(vprop, need, inputs, out):
    for inp in inputs:
        vprop.add_need(inp, ANYTHING)
    vprop.add_value(out, need, ANYTHING)


#######################
# DeadDataElimination #
#######################


class DeadDataElimination(Partializable):
    """Eliminate expressions that compute unretrieved data."""

    def __init__(self, resources=None):
        """Initialize a DeadDataElimination."""
        self.resources = resources
        self.name = "dde"

    def make_dead(self, node):
        """Create a dead version of the node."""
        a = node.abstract
        val = DEAD
        if isinstance(a, abstract.AbstractScalar):
            if a.xtype() == xtype.EnvType:
                val = newenv
            elif a.xtype() == xtype.UniverseType:
                return None
        repl = Constant(val)
        repl.abstract = node.abstract
        return repl

    def __call__(self, root):
        """Apply dead data elimination."""
        args = dict(opt=self, node=None, manager=root.manager, profile=False)
        with tracer("opt", **args) as tr:
            tr.set_results(success=False, **args)
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
                if repl is not None:
                    mng.replace(node, repl)
            tracer().emit_success(**args, new_node=None)
            return False  # Pretend there are no changes, for now


__all__ = ["ValuePropagator", "DeadDataElimination"]
