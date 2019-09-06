"""Implementations of the primitives' gradients.

Each primitive is associated to an augmented function, which returns a pair of
the (augmented) original primitive's output and a backpropagator function.
"""

from .. import operations
from ..abstract import AbstractFunction, GraphFunction
from ..composite import zeros_like
from ..debug.label import short_labeler, short_relation_symbols as syms
from ..info import About, NamedDebugInfo
from ..ir import Constant, Graph, MetaGraph, clone, manage
from ..opt.lib import Xs
from ..parser import operations_ns
from ..pipeline import standard_pipeline
from ..utils import InternalInferenceError, Registry, newenv
from . import ops as primops
from .py_implementations import (
    J,
    Jinv,
    array_reduce,
    array_to_scalar,
    casttag,
    conv2d_input_grad,
    conv2d_weight_grad,
    distribute,
    dot,
    invert_permutation,
    reshape,
    scalar_add,
    scalar_cast,
    scalar_div,
    scalar_eq,
    scalar_gt,
    scalar_log,
    scalar_mul,
    scalar_pow,
    scalar_sub,
    scalar_to_array,
    scalar_usub,
    shape,
    switch,
    tagged,
    transpose,
    tuple_setitem,
    typeof,
    unsafe_static_cast,
)

parse = standard_pipeline \
    .select('resources', 'parse') \
    .make_transformer('input', 'graph')


_flags = {'ignore_values': True, 'core': True, 'reference': True}


_is_mktuple = ((operations.resolve, operations_ns, 'make_tuple'), Xs)
_is_raise = (primops.raise_, Xs)


def bprop_to_augm(prim, fn, flags):
    """Given a function for the bprop, make the augmented function."""
    info = NamedDebugInfo(prim=prim, name=prim.name)

    bprop = clone(parse(fn))
    bprop.flags.update(_flags)
    bprop.debug.name = None
    bprop.debug.about = About(info, 'grad_bprop')  # type: ignore
    if bprop.output.match(_is_raise):
        pass
    elif bprop.output.match(_is_mktuple):
        bprop.output = bprop.apply(
            primops.make_tuple,
            newenv,
            *bprop.output.inputs[1:]
        )
    else:
        raise InternalInferenceError(
            f'The backpropagator for {prim} is not defined properly. '
            f'It should return a tuple literal.',
            refs=[bprop.return_]
        )

    *args, out_param, dout = bprop.parameters

    with About(info, 'grad_fprop'):
        outer = Graph()
        outer.flags.update(_flags)
        outer.flags.update(flags)
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


def register_bprop(prim, **flags):
    """Register an augmented function for prim, given a backpropagator."""
    def deco(fn):
        g = bprop_to_augm(prim, fn, flags)
        return register(prim)(g)
    return deco


def register_augm(prim):
    """Register an augmented function for prim."""
    def deco(fn):
        g = parse(fn)
        for g2 in manage(g, weak=True).graphs:
            name = short_labeler.name(g2)
            name = name.replace('__fprop__', syms['grad_fprop'])
            g2.debug.name = name.replace('__bprop__', syms['grad_bprop'])
            g2.flags.update(_flags)
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


@register_bprop(primops.scalar_exp)
def bprop_scalar_exp(x, out, dout):
    """Backpropagator for primitive `scalar_exp`."""
    return (dout * out,)


@register_bprop(primops.scalar_log)
def bprop_scalar_log(x, out, dout):
    """Backpropagator for primitive `scalar_log`."""
    return (dout / x,)


@register_bprop(primops.scalar_tanh)
def bprop_scalar_tanh(x, out, dout):
    """Backpropagator for primitive `scalar_tanh`."""
    return (dout - dout * out * out,)


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


@register_bprop(primops.scalar_eq)
def bprop_scalar_eq(x, y, out, dout):
    """Backpropagator for primitive `scalar_eq`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_ge)
def bprop_scalar_ge(x, y, out, dout):
    """Backpropagator for primitive `scalar_ge`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_le)
def bprop_scalar_le(x, y, out, dout):
    """Backpropagator for primitive `scalar_le`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_max)
def bprop_scalar_max(x, y, out, dout):
    """Backpropagator for primitive `scalar_max`."""
    ret = switch(scalar_eq(x, y), (dout, dout),
                 switch(scalar_gt(x, y), (dout, zeros_like(y)),
                        (zeros_like(x), dout)))
    return (ret[0], ret[1])


@register_bprop(primops.scalar_cast)
def bprop_scalar_cast(x, t, out, dout):
    """Backpropagator for primitive `scalar_cast`."""
    return (scalar_cast(dout, typeof(x)), t)


@register_bprop(primops.tuple_getitem, ignore_values=False)
def bprop_tuple_getitem(data, idx, out, dout):
    """Backpropagator for primitive `tuple_getitem`."""
    return (tuple_setitem(zeros_like(data), idx, dout),
            zeros_like(idx))


@register_bprop(primops.identity)
def bprop_identity(x, out, dout):
    """Backpropagator for primitive `identity`."""
    return (dout,)


@register_bprop(primops.scalar_to_array)
def bprop_scalar_to_array(x, t, out, dout):
    """Backpropagator for primitive `scalar_to_array`."""
    return (array_to_scalar(dout),)


@register_bprop(primops.array_to_scalar)
def bprop_array_to_scalar(x, out, dout):
    """Backpropagator for primitive `array_to_scalar`."""
    return (scalar_to_array(dout, typeof(x)),)


@register_bprop(primops.dot)
def bprop_dot(x, y, out, dout):
    """Backpropagator for primitive `dot`."""
    return (dot(dout, transpose(y, (1, 0))),
            dot(transpose(x, (1, 0)), dout))


@register_bprop(primops.reshape, ignore_values=False)
def bprop_reshape(xs, shp, out, dout):
    """Backpropagator for primitive `reshape`."""
    return (reshape(dout, shape(xs)),
            zeros_like(shp))


@register_bprop(primops.transpose)
def bprop_transpose(xs, perm, out, dout):
    """Backpropagator for primitive `transpose`."""
    return (transpose(dout, invert_permutation(perm)),
            zeros_like(perm))


@register_bprop(primops.distribute)
def bprop_distribute(arr, shp, out, dout):
    """Backpropagator for primitive `distribute`."""
    return (array_reduce(scalar_add, dout, shape(arr)),
            zeros_like(shp))


@register_bprop(primops.shape)
def bprop_shape(arr, out, dout):
    """Backpropagator for primitive `shape`."""
    return (zeros_like(arr),)


@register_bprop(primops.broadcast_shape)
def bprop_broadcast_shape(shp1, shp2, out, dout):
    """Backpropagator for primitive `broadcast_shape`."""
    return (zeros_like(shp1), zeros_like(shp2))


@register_bprop(primops.J)
def bprop_J(x, out, dout):
    """Backpropagator for primitive `J`."""
    return (Jinv(dout),)


@register_bprop(primops.Jinv)
def bprop_Jinv(x, out, dout):
    """Backpropagator for primitive `Jinv`."""
    return (J(dout),)


@register_bprop(primops.conv2d, ignore_values=False)
def bprop_conv2d(input, weight, stride, padding, dilation, groups, out, dout):
    """Backpropagator for `conv2d`."""
    gI = conv2d_input_grad(shape(input), weight, dout, stride, padding,
                           dilation, groups)
    gW = conv2d_weight_grad(input, shape(weight), dout, stride, padding,
                            dilation, groups)
    return (gI, gW,
            zeros_like(stride), zeros_like(padding), zeros_like(dilation),
            zeros_like(groups))


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


@register_bprop(primops.hastag, ignore_values=False)
def bprop_hastag(x, t, out, dout):
    """Backpropagator for primitive `hastag`."""
    return (zeros_like(x), zeros_like(t))


@register_bprop(primops.casttag, ignore_values=False)
def bprop_casttag(x, t, out, dout):
    """Backpropagator for primitive `casttag`."""
    return (unsafe_static_cast(tagged(dout, t), typeof(x)),
            zeros_like(t))


@register_bprop(primops.tagged, ignore_values=False)
def bprop_tagged(x, t, out, dout):
    """Backpropagator for primitive `tagged`."""
    return (casttag(dout, t), zeros_like(t))


@register_bprop(primops.raise_)
def bprop_raise_(x, out, dout):
    """Backpropagator for primitive `raise_`."""
    raise x


@register_bprop(primops.exception)
def bprop_exception(x, out, dout):
    """Backpropagator for primitive `exception`."""
    return x,


class MakeTupleGradient(MetaGraph):
    """Generate the gradient graph for make_tuple."""

    def generate_graph(self, args):
        """Generate the gradient graph."""
        g = Graph()
        g.debug.name = f'{syms["grad_fprop"]}make_tuple_{len(args)}'

        params = [g.add_parameter() for t in args]
        jinv_params = [g.apply(primops.Jinv, p) for p in params]
        tup = g.apply(primops.make_tuple, *jinv_params)
        out = g.apply(primops.J, tup)

        b = Graph()
        b.debug.name = f'{syms["grad_bprop"]}make_tuple_{len(args)}'
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


class ArrayMapGradient(MetaGraph):
    """Generate the gradient graph for array_map.

    Sketch of the transform:

        array_map(f, xs, ys, ...) =>

        def fprop_array_map(jf, jxs, jys, ...):
            f, xs, ys, ... = Jinv(jf), Jinv(jxs), Jinv(jys), ...
            ret = array_map(f, xs, ys, ...)

            def bprop_array_map(dout):
                df = newenv
                f_dxs = lambda d, jx, jy, ...: jf(jx, jy, ...)[1](d)[1]
                dxs = array_map(f_dxs, dout, jxs, jys, ...)
                f_dys = lambda d, jx, jy, ...: jf(jx, jy, ...)[1](d)[2]
                dys = array_map(f_dys, dout, jxs, jys, ...)
                ...
                return df, dxs, dys, ...

            return ret, bprop_array_map
    """

    def generate_graph(self, absargs):
        """Generate the gradient graph."""
        g = Graph()
        nargs = len(absargs) - 1
        params = [g.add_parameter() for _ in range(nargs + 1)]
        jf, *jargs = params
        f, *args = [g.apply(primops.Jinv, p) for p in params]
        ret = g.apply(primops.array_map, f, *args)

        b = Graph()
        dout = b.add_parameter()

        results = []

        for i in range(nargs):
            func = Graph()
            fparams = [func.add_parameter() for _ in range(nargs + 1)]
            fparams[0].debug.name = f'{syms["grad_sens"]}out'
            fjparams = [func.apply(primops.J, p) for p in fparams]
            call = func.apply(jf, *fjparams[1:])
            bprop = func.apply(primops.tuple_getitem, call, 1)
            sens = func.apply(bprop, fparams[0])
            func.output = func.apply(primops.tuple_getitem, sens, i + 1)
            result = b.apply(primops.array_map, func, dout, *args)
            results.append(result)

        b.output = b.apply(primops.make_tuple, newenv, newenv, *results)

        ret = g.apply(primops.J, ret)
        g.output = g.apply(primops.make_tuple, ret, b)

        b.flags.update(_flags)
        g.flags.update(_flags)

        return g


register(primops.array_map)(ArrayMapGradient(name='array_map_gradient'))


def bprop_sum(fn, xs, shp, out, dout):  # pragma: no cover
    """Backpropagator for sum(xs) = array_reduce(scalar_add, xs, shp)."""
    return (newenv,
            distribute(dout, shape(xs)),
            zeros_like(shp))


class ArrayReduceGradient(MetaGraph):
    """Generate the gradient graph for array_reduce.

    For the time being, the gradient of array_reduce is only supported
    over the `scalar_add` operation (sum, basically).
    """

    def generate_graph(self, args):
        """Generate the gradient graph."""
        jf, jarr, jshp = args
        assert isinstance(jf, AbstractFunction)
        fn = jf.get_unique()
        assert isinstance(fn, GraphFunction) and fn.graph.parent is None
        assert fn.graph.transforms['primal'] is primops.scalar_add
        return bprop_to_augm(primops.array_reduce, bprop_sum, {})


register(primops.array_reduce)(
    ArrayReduceGradient(name='array_reduce_gradient')
)
