"""Implementations of primitives as graphs."""


from dataclasses import dataclass
from functools import reduce

from .dtype import Array, Object, Int, UInt, Float, Number, Bool, Tuple, \
    List, Class, EnvType, ismyiatype
from .hypermap import HyperMap
from .infer import Inferrer, GraphInferrer, MyiaTypeError
from .info import About
from .ir import Graph, MetaGraph, MultitypeGraph, Constant
from .prim import ops as P
from .prim.py_implementations import \
    array_map, bool_not, bool_eq, hastype, distribute, shape, \
    broadcast_shape, switch, identity, bool_and, typeof, scalar_cast, \
    scalar_add, scalar_exp, scalar_log, scalar_sin, scalar_cos, scalar_tan, \
    scalar_div, scalar_to_array, env_add
from .utils import newenv


def core(fn):
    """Wrap a graph that defines a core Myia function.

    The resulting graph is given the following flags:
        core: Indicates that this is a core function (only informative at
            the moment).
        flatten_inference: Tells the InferenceEngine to infer through this
            function as if it was inlined, disregarding depth limitations.
    """
    fn._myia_flags = {
        # This is a function defined in Myia's core
        'core': True,
        # Inference should not broaden context when entering this function
        'flatten_inference': True,
    }
    return fn


@core
def arrayable_binary(mname, x, y):
    """Define a binary function that upcasts its arguments to Array.

    If either x or y is an Array, the other is converted to an array using
    the `to_array` core function.

    Arguments:
        mname: The method name to call on x, e.g. `__add__`
        x: The first argument.
        y: The second argument.
    """
    x_isarr = hastype(x, Array)
    y_isarr = hastype(y, Array)

    x_cvt = bool_and(y_isarr, bool_not(x_isarr))
    y_cvt = bool_and(x_isarr, bool_not(y_isarr))

    x_tfm = switch(x_cvt, to_array, identity)
    y_tfm = switch(y_cvt, to_array, identity)

    x = x_tfm(x)
    y = y_tfm(y)

    return getattr(x, mname)(y)


@core
def add(x, y):
    """Implementation of `add`."""
    return arrayable_binary('__add__', x, y)


@core
def sub(x, y):
    """Implementation of `sub`."""
    return arrayable_binary('__sub__', x, y)


@core
def mul(x, y):
    """Implementation of `mul`."""
    return arrayable_binary('__mul__', x, y)


@core
def truediv(x, y):
    """Implementation of `truediv`."""
    return arrayable_binary('__truediv__', x, y)


@core
def floordiv(x, y):
    """Implementation of `floordiv`."""
    return arrayable_binary('__floordiv__', x, y)


@core
def mod(x, y):
    """Implementation of `mod`."""
    return arrayable_binary('__mod__', x, y)


@core
def pow(x, y):
    """Implementation of `pow`."""
    return arrayable_binary('__pow__', x, y)


@core
def floor(x):
    """Implementation of `floor`."""
    return x.__floor__()


@core
def trunc(x):
    """Implementation of `trunc`."""
    return x.__trunc__()


@core
def uadd(x):
    """Implementation of `uadd`."""
    return x.__pos__()


@core
def usub(x):
    """Implementation of `usub`."""
    return x.__neg__()


@core
def scalar_truediv(x, y):
    """Implementation of `scalar_truediv`."""
    return x.__truediv__(y)


@core
def scalar_floordiv(x, y):
    """Implementation of `scalar_floordiv`."""
    return x.__floordiv__(y)


@core
def bool_ne(x, y):
    """Implementation of `bool_ne`."""
    return bool_not(bool_eq(x, y))


exp = MultitypeGraph('exp')
log = MultitypeGraph('log')
sin = MultitypeGraph('sin')
cos = MultitypeGraph('cos')
tan = MultitypeGraph('tan')


@exp.register(Number)
@core
def _exp(x):
    return scalar_exp(x)


@log.register(Number)
@core
def _log(x):
    return scalar_log(x)


@sin.register(Number)
@core
def _sin(x):
    return scalar_sin(x)


@cos.register(Number)
@core
def _cos(x):
    return scalar_cos(x)


@tan.register(Number)
@core
def _tan(x):
    return scalar_tan(x)


@core
def eq(x, y):
    """Implementation of `eq`."""
    return arrayable_binary('__eq__', x, y)


@core
def lt(x, y):
    """Implementation of `lt`."""
    return arrayable_binary('__lt__', x, y)


@core
def gt(x, y):
    """Implementation of `gt`."""
    return arrayable_binary('__gt__', x, y)


@core
def ne(x, y):
    """Implementation of `ne`."""
    return arrayable_binary('__ne__', x, y)


@core
def le(x, y):
    """Implementation of `le`."""
    return arrayable_binary('__le__', x, y)


@core
def ge(x, y):
    """Implementation of `ge`."""
    return arrayable_binary('__ge__', x, y)


@core
def bool(x):
    """Implementation of `bool`."""
    return x.__bool__()


@core
def not_(x):
    """Implementation of `not`."""
    return bool_not(bool(x))


@core
def and_(x, y):
    """Implementation of `and` (`&`)."""
    return x.__and__(y)


@core
def or_(x, y):
    """Implementation of `or` (`|`)."""
    return x.__or__(y)


@core
def matmul(x, y):
    """Implementation of `matmul` (`@`)."""
    return x.__matmul__(y)


##################
# Scalar methods #
##################


@core
def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


# The parser/inferrer don't like when those are defined inline.
ui8 = UInt[8]
ui16 = UInt[16]
i8 = Int[8]
i16 = Int[16]
f32 = Float[32]
f64 = Float[64]


@core
def int_floordiv(x, y):
    """Implementation of `int_floordiv`."""
    if (x <= 0) == (y <= 0):
        return scalar_div(x, y)
    else:
        return scalar_div(x, y) - 1


@core
def int_truediv(x, y):
    """Implementation of `int_truediv`."""
    if hastype(x, typeof(y)):
        if (hastype(x, i8) or hastype(x, ui8) or
                hastype(x, i16) or hastype(x, ui16)):
            return scalar_cast(x, f32) / scalar_cast(y, f32)
        return scalar_cast(x, f64) / scalar_cast(y, f64)
    else:
        # This branch is only here to trigger a type check error.
        return x / y


@core
def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0


@core
def float_floordiv(x, y):
    """Implementation of `float_floordiv`."""
    return floor(x / y)


#############
# Iteration #
#############


@core
def _len(data):
    """Implementation of `len`."""
    return data.__len__()


@core
def getitem(data, item):
    """Implementation of `getitem`."""
    return data.__getitem__(item)


@core
def setitem(data, item, value):
    """Implementation of `setitem`."""
    return data.__setitem__(item, value)


@core
def iter(xs):
    """Implementation of `iter`."""
    return xs.__myia_iter__()


@core
def next(it):
    """Implementation of `next`."""
    return it.__myia_next__()


@core
def hasnext(it):
    """Implementation of `hasnext`."""
    return it.__myia_hasnext__()


@dataclass(frozen=True)
class SequenceIterator:
    """Iterator to use for sequences like List, Array."""

    idx: Int
    seq: Object

    @core
    def __myia_hasnext__(self):
        """Whether the index is past the length of the sequence."""
        return self.idx < len(self.seq)

    @core
    def __myia_next__(self):
        """Return the next element and a new iterator."""
        return self.seq[self.idx], SequenceIterator(self.idx + 1, self.seq)


@core
def list_iter(xs):
    """Iterator for List."""
    return SequenceIterator(0, xs)


@core
def array_iter(xs):
    """Iterator for Array."""
    return SequenceIterator(0, xs)


@core
def tuple_next(xs):
    """Next tuple."""
    return xs[0], tail(xs)


@core
def tuple_hasnext(xs):
    """Whether the tuple is empty or not."""
    return len(xs) > 0


class Tail(MetaGraph):
    """Implementation of tail."""

    def specialize_from_types(self, types):
        """Generate tail specialized for the given Tuple type.

        tail(x) generates make_tuple(x[1], x[2], ...)
        """
        if len(types) != 1:
            raise MyiaTypeError('tail takes one argument')
        t, = types
        if not ismyiatype(t, Tuple):
            raise MyiaTypeError('tail requires a Tuple')
        if len(t.elements) == 0:
            raise MyiaTypeError('tail requires a non-empty Tuple')
        g = Graph()
        g.flags['core'] = True
        g.flags['flatten_inference'] = True
        tup = g.add_parameter()
        tup.debug.name = "tup"
        elems = [g.apply(P.tuple_getitem, tup, i)
                 for i in range(1, len(t.elements))]
        g.output = g.apply(P.make_tuple, *elems)
        return g


tail = Tail('tail')


#################
# Array methods #
#################


@core
def broadcastable_binary(op, xs, ys):
    """Define a binary elementwise function that broadcasts its arguments.

    This operation broadcasts xs and ys to the minimal compatible shape.

    Arguments:
        op: The operation to apply pairwise to every element of xs and ys.
        xs: The first Array.
        ys: The second Array.
    """
    shp = broadcast_shape(shape(xs), shape(ys))
    xs = distribute(xs, shp)
    ys = distribute(ys, shp)
    res = array_map(op, xs, ys)
    return res


@core
def to_array(x):
    """Implementation of `to_array`."""
    return x.__myia_to_array__()


@core
def array_add(xs, ys):
    """Implementation of `array_add`."""
    return broadcastable_binary(add, xs, ys)


@core
def array_sub(xs, ys):
    """Implementation of `array_sub`."""
    return broadcastable_binary(sub, xs, ys)


@core
def array_mul(xs, ys):
    """Implementation of `array_mul`."""
    return broadcastable_binary(mul, xs, ys)


@core
def array_truediv(xs, ys):
    """Implementation of `array_truediv`."""
    return broadcastable_binary(truediv, xs, ys)


@core
def array_floordiv(xs, ys):
    """Implementation of `array_floordiv`."""
    return broadcastable_binary(floordiv, xs, ys)


@core
def array_mod(xs, ys):
    """Implementation of `array_mod`."""
    return broadcastable_binary(mod, xs, ys)


@core
def array_pow(xs, ys):
    """Implementation of `array_pow`."""
    return broadcastable_binary(pow, xs, ys)


@core
def array_floor(xs):
    """Implementation of `array_floor`."""
    return array_map(floor, xs)


@core
def array_trunc(xs):
    """Implementation of `array_trunc`."""
    return array_map(trunc, xs)


@core
def array_uadd(xs):
    """Implementation of `array_uadd`."""
    return array_map(uadd, xs)


@core
def array_usub(xs):
    """Implementation of `array_usub`."""
    return array_map(usub, xs)


@exp.register(Array)
@core
def array_exp(xs):
    """Implementation of `array_exp`."""
    return array_map(scalar_exp, xs)


@log.register(Array)
@core
def array_log(xs):
    """Implementation of `array_log`."""
    return array_map(scalar_log, xs)


@sin.register(Array)
@core
def array_sin(xs):
    """Implementation of `array_sin`."""
    return array_map(scalar_sin, xs)


@cos.register(Array)
@core
def array_cos(xs):
    """Implementation of `array_cos`."""
    return array_map(scalar_cos, xs)


@tan.register(Array)
@core
def array_tan(xs):
    """Implementation of `array_tan`."""
    return array_map(scalar_tan, xs)


@core
def array_eq(xs, ys):
    """Implementation of `array_eq`."""
    return broadcastable_binary(eq, xs, ys)


@core
def array_lt(xs, ys):
    """Implementation of `array_lt`."""
    return broadcastable_binary(lt, xs, ys)


@core
def array_gt(xs, ys):
    """Implementation of `array_gt`."""
    return broadcastable_binary(gt, xs, ys)


@core
def array_ne(xs, ys):
    """Implementation of `array_ne`."""
    return broadcastable_binary(ne, xs, ys)


@core
def array_le(xs, ys):
    """Implementation of `array_le`."""
    return broadcastable_binary(le, xs, ys)


@core
def array_ge(xs, ys):
    """Implementation of `array_ge`."""
    return broadcastable_binary(ge, xs, ys)


_leaf_add = MultitypeGraph('hyper_add')


@_leaf_add.register(Number, Number)
@core
def _scalar_add(x, y):
    return scalar_add(x, y)


@_leaf_add.register(EnvType, EnvType)
@core
def _sm_add(x, y):
    return env_add(x, y)


hyper_add = HyperMap(fn_leaf=_leaf_add)


_leaf_zeros_like = MultitypeGraph('zeros_like')


@_leaf_zeros_like.register(Inferrer)
@core
def _inferrer_zero(_):
    return newenv


@_leaf_zeros_like.register(Bool)
@core
def _bool_zero(_):
    return False


@_leaf_zeros_like.register(Number)
@core
def _scalar_zero(x):
    return scalar_cast(0, typeof(x))


@_leaf_zeros_like.register(Array)
@core
def _array_zero(xs):
    scalar_zero = scalar_cast(0, typeof(xs).elements)
    return distribute(to_array(scalar_zero), shape(xs))


zeros_like = HyperMap(
    nonleaf=(Tuple, List, Class),
    fn_leaf=_leaf_zeros_like
)


@core
def list_reduce(fn, lst, dftl):
    """Implementation of list_reduce."""
    res = dftl
    i = 0
    while i < len(lst):
        res = fn(res, lst[i])
        i = i + 1
    return res


class ListMap(MetaGraph):
    """Implementation of list_map."""

    def specialize_from_types(self, types):
        """Return a graph for the number of lists."""
        if len(types) < 2:
            raise MyiaTypeError('list_map takes at least two arguments')

        g = Graph()
        g.flags['core'] = True
        g.flags['flatten_inference'] = True
        g.debug.name = 'list_map'
        fn = g.add_parameter()
        lists = [g.add_parameter() for _ in types[1:]]
        iters = [g.apply(list_iter, l) for l in lists]
        nexts = [g.apply(next, it) for it in iters]
        values = [g.apply(P.tuple_getitem, n, 0) for n in nexts]
        iters = [g.apply(P.tuple_getitem, n, 1) for n in nexts]
        resl = g.apply(P.make_list, g.apply(fn, *values))

        gnext = Graph()
        gnext.debug.name = 'body'
        gcond = Graph()
        gcond.debug.name = 'cond'

        def make_cond(g):
            fn = g.add_parameter()
            resl = g.add_parameter()
            iters = [g.add_parameter() for _ in lists]
            hasnexts = [g.apply(hasnext, it) for it in iters]
            cond = reduce(lambda a, b: g.apply(P.bool_and, a, b), hasnexts)
            gtrue = Graph()
            gtrue.debug.name = 'ftrue'
            gtrue.flags['core'] = True
            gtrue.flags['flatten_inference'] = True
            gtrue.output = gtrue.apply(gnext, fn, resl, *iters)
            gfalse = Graph()
            gfalse.debug.name = 'ffalse'
            gfalse.flags['core'] = True
            gfalse.flags['flatten_inference'] = True
            gfalse.output = resl
            g.output = g.apply(g.apply(P.switch, cond, gtrue, gfalse))

        def make_next(g):
            fn = g.add_parameter()
            resl = g.add_parameter()
            iters = [g.add_parameter() for _ in lists]
            nexts = [g.apply(next, it) for it in iters]
            values = [g.apply(P.tuple_getitem, n, 0) for n in nexts]
            iters = [g.apply(P.tuple_getitem, n, 1) for n in nexts]
            resl = g.apply(P.list_append, resl, g.apply(fn, *values))
            g.output = g.apply(gcond, fn, resl, *iters)

        make_cond(gcond)
        make_next(gnext)
        g.output = g.apply(gcond, fn, resl, *iters)

        return g

    def __call__(self, fn, *lists):
        """Python implementation of list_map."""
        from .prim.py_implementations import list_map
        return list_map(fn, *lists)


list_map = ListMap('list_map')


@core
def _cast_helper(x, model):
    t = typeof(model)
    if hastype(model, Array):
        return scalar_to_array(scalar_cast(x, t.elements))
    else:
        return scalar_cast(x, t)


class GradOperation(MetaGraph):
    """Implements the grad(f) operation.

    grad(f)(x, ...) returns df(x, ...)/dx. Derivatives of other inputs are
    thrown out.

    TODO: This currently will not work on primitives, but it is an easy fix.
    We just need to know how many parameters f takes.
    """

    def make_gf(self, jf, orig_params,
                dbg, sens_param=False, get_all=False,
                apply_j=False):
        """Make the graph for the grad."""
        with About(dbg, 'grad'):
            df = Graph()

        if apply_j:
            jf = df.apply(P.J, jf)

        params = []
        for orig_p in orig_params:
            with About(orig_p.debug, 'grad'):
                params.append(df.add_parameter())

        jparams = [df.apply(P.J, p) for p in params]
        app = df.apply(jf, *jparams)
        out = df.apply(P.Jinv, df.apply(P.tuple_getitem, app, 0))
        bprop = df.apply(P.tuple_getitem, app, 1)

        if sens_param:
            bprop_arg = df.add_parameter()
        else:
            bprop_arg = df.apply(_cast_helper, 1, out)

        bapp = df.apply(bprop, bprop_arg)
        if get_all:
            df.output = df.apply(tail, bapp)
        else:
            df.output = df.apply(P.tuple_getitem, bapp, 1)
        return df

    def specialize_from_types(self, types):
        """Generate the graph."""
        ft, = types
        assert isinstance(ft, GraphInferrer)
        g = ft._graph
        assert isinstance(g, Graph)

        dfbuilder = Graph()
        dfbuilder.debug.name = f"grad{len(g.parameters)}"

        with About(g.debug, 'copy'):
            fn = dfbuilder.add_parameter()

        with About(g.debug, 'grad_fprop'):
            jf = dfbuilder.apply(P.J, fn)

        df = self.make_gf(jf, g.parameters, g.debug)

        dfbuilder.output = Constant(df)

        return dfbuilder


grad = GradOperation('grad')
