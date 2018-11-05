"""Implementations of the primitives' gradients.

Each primitive is associated to an augmented function, which returns a pair of
the (augmented) original primitive's output and a backpropagator function.
"""

from ..api import standard_pipeline
from ..composite import zeros_like
from ..info import NamedDebugInfo, About
from ..ir import Constant, Graph, manage, clone, MetaGraph
from ..utils import Registry, newenv

from . import ops as primops
from .py_implementations import \
    Jinv, J, \
    scalar_mul, scalar_div, scalar_sub, scalar_usub, scalar_log, scalar_pow, \
    tuple_setitem, switch


parse = standard_pipeline \
    .select('parse') \
    .make_transformer('input', 'graph')


_flags = {
    'flatten_inference': True,
}


def bprop_to_augm(prim, fn):
    """Given a function for the bprop, make the augmented function."""
    info = NamedDebugInfo(prim=prim, name=prim.name)

    bprop = clone(parse(fn))
    bprop.flags.update(_flags)
    bprop.debug.name = None
    bprop.debug.about = About(info, 'grad_bprop')  # type: ignore
    bprop.output = bprop.apply(
        primops.make_tuple,
        newenv,
        *bprop.output.inputs[1:]
    )

    *args, out_param, dout = bprop.parameters

    with About(info, 'grad_fprop'):
        outer = Graph()
        outer.flags.update(_flags)
        outer.transforms['primal'] = prim
        outer.output = Constant(None)

    mng = manage(bprop, outer)

    transf_args = []
    for p in args:
        with About(p.debug, 'grad_fprop'):
            outer_p = outer.add_parameter()
        with About(p.debug, 'equiv'):
            transf_p = outer.apply(primops.Jinv, outer_p)
        mng.replace(p, transf_p)
        transf_args.append(transf_p)

    with About(out_param.debug, 'equiv'):
        out_value = outer.apply(prim, *transf_args)

    mng.replace(out_param, out_value)

    with About(out_param.debug, 'grad_sens'):
        new_dout = bprop.add_parameter()
        mng.replace(dout, new_dout)
        # We remove all parameters except new_dout
        bprop.parameters = [new_dout]

    result = outer.apply(primops.J, out_value)
    outer.output = outer.apply(
        primops.make_tuple,
        result,
        bprop
    )
    return clone(outer)


augmented_graphs = Registry()
register = augmented_graphs.register


def register_bprop(prim):
    """Register an augmented function for prim, given a backpropagator."""
    def deco(fn):
        g = bprop_to_augm(prim, fn)
        return register(prim)(g)
    return deco


def register_augm(prim):
    """Register an augmented function for prim."""
    from ..debug.label import short_labeler, short_relation_symbols as syms

    def deco(fn):
        g = parse(fn)
        for g2 in manage(g, weak=True).graphs:
            name = short_labeler.name(g2)
            name = name.replace('__fprop__', syms['grad_fprop'])
            g2.debug.name = name.replace('__bprop__', syms['grad_bprop'])
        g.transforms['primal'] = prim
        return register(prim)(g)
    return deco


@register_bprop(primops.scalar_add)
def bprop_scalar_add(x, y, out, dout):
    """Backpropagator for primitive `scalar_add`."""
    return (dout, dout)


@register_bprop(primops.scalar_sub)
def bprop_scalar_sub(x, y, out, dout):
    """Backpropagator for primitive `scalar_sub`."""
    return (dout, scalar_usub(dout))


@register_bprop(primops.scalar_mul)
def bprop_scalar_mul(x, y, out, dout):
    """Backpropagator for primitive `scalar_mul`."""
    return (scalar_mul(dout, y), scalar_mul(dout, x))


@register_bprop(primops.scalar_div)
def bprop_scalar_div(x, y, out, dout):
    """Backpropagator for primitive `scalar_div`."""
    return (scalar_div(dout, y),
            scalar_mul(scalar_usub(dout), scalar_div(out, y)))


@register_bprop(primops.scalar_pow)
def bprop_scalar_pow(x, y, out, dout):
    """Backpropagator for primitive `scalar_pow`."""
    return (scalar_mul(dout, scalar_mul(y, scalar_pow(x, scalar_sub(y, 1)))),
            scalar_mul(dout, scalar_mul(scalar_log(x), out)))


@register_bprop(primops.scalar_uadd)
def bprop_scalar_uadd(x, out, dout):
    """Backpropagator for primitive `scalar_uadd`."""
    return (dout,)


@register_bprop(primops.scalar_usub)
def bprop_scalar_usub(x, out, dout):
    """Backpropagator for primitive `scalar_usub`."""
    return (scalar_usub(dout),)


@register_bprop(primops.scalar_gt)
def bprop_scalar_gt(x, y, out, dout):
    """Backpropagator for primitive `scalar_gt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_lt)
def bprop_scalar_lt(x, y, out, dout):
    """Backpropagator for primitive `scalar_lt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_ge)
def bprop_scalar_ge(x, y, out, dout):
    """Backpropagator for primitive `scalar_ge`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_le)
def bprop_scalar_le(x, y, out, dout):
    """Backpropagator for primitive `scalar_le`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.tuple_getitem)
def bprop_tuple_getitem(data, idx, out, dout):
    """Backpropagator for primitive `tuple_getitem`."""
    return (tuple_setitem(zeros_like(data), idx, dout),
            zeros_like(idx))


@register_bprop(primops.J)
def bprop_J(x, out, dout):
    """Backpropagator for primitive `J`."""
    return (Jinv(dout),)


@register_bprop(primops.Jinv)
def bprop_Jinv(x, out, dout):
    """Backpropagator for primitive `Jinv`."""
    return (J(dout),)


@register_augm(primops.switch)
def __fprop__switch(jcond, jtb, jfb):
    """Backpropagator for primitive `switch`."""
    cond = Jinv(jcond)
    rval = switch(cond, jtb, jfb)

    def __bprop__switch(dout):
        tb = Jinv(jtb)
        fb = Jinv(jfb)
        return (newenv,
                zeros_like(cond),
                switch(cond, dout, zeros_like(fb)),
                switch(cond, zeros_like(tb), dout))
    return rval, __bprop__switch


class MakeTupleGradient(MetaGraph):
    """Generate the gradient graph for make_tuple."""

    def specialize_from_types(self, types):
        """Generate the gradient graph."""
        g = Graph()

        params = [g.add_parameter() for t in types]
        jinv_params = [g.apply(primops.Jinv, p) for p in params]
        tup = g.apply(primops.make_tuple, *jinv_params)
        out = g.apply(primops.J, tup)

        b = Graph()
        dout = b.add_parameter()
        grads = [b.apply(primops.tuple_getitem, dout, i)
                 for i, p in enumerate(params)]
        b.output = b.apply(primops.make_tuple, newenv, *grads)

        g.output = g.apply(primops.make_tuple, out, b)
        g.transforms['primal'] = primops.make_tuple

        b.flags.update(_flags)
        g.flags.update(_flags)

        return g


register(primops.make_tuple)(MakeTupleGradient(name='make_tuple_gradient'))
