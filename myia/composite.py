"""Implementations of primitives as graphs."""


from dataclasses import dataclass
from functools import reduce

from .abstract import AbstractArray, SHAPE, ANYTHING, MyiaShapeError, \
    AbstractFunction, GraphFunction, AbstractList, AbstractTuple, \
    AbstractClass, build_value, AbstractError
from .debug.label import short_labeler
from .dtype import Array, Number, Bool, \
    EnvType, u8, u16, i8, i16, f32, f64
from .hypermap import HyperMap, hyper_map
from .abstract import MyiaTypeError, broaden
from .info import About
from .ir import Graph, MetaGraph, MultitypeGraph, Constant
from .prim import ops as P
from .prim.py_implementations import \
    array_map, bool_not, bool_eq, hastype, distribute, shape, \
    broadcast_shape, typeof, scalar_cast, scalar_add, scalar_exp, \
    scalar_log, scalar_sin, scalar_cos, scalar_tan, scalar_div, \
    scalar_to_array, env_add, scalar_tanh, py_registry, array_reduce
from .utils import newenv


def core(fn=None, **flags):
    """Wrap a graph that defines a core Myia function.

    The following flags can be set:
        core: (default: True) Indicates that this is a core function
            (only informative at the moment).
        ignore_values: (default: False) Make the inferrer ignore argument
            values for the parameters (leads to less specialization).
    """
    flags = {
        # This is a function defined in Myia's core
        'core': True,
        **flags,
    }

    def deco(fn):
        fn._myia_flags = flags
        return fn

    if fn is None:
        return deco
    else:
        return deco(fn)


class Elemwise(MetaGraph):
    """Generate a graph for an elemwise operation.

    * If any argument is an array:
      * All scalar arguments are converted to arrays using scalar_to_array.
      * The arguments are all broadcasted and array_map is called on them.
    * Otherwise, we return getattr(arg1, mname)(arg2, ...)
    """

    def __init__(self, mname, scalar_op=None, infer_value=False, name=None):
        """Initialize Elemwise."""
        super().__init__(name or mname)
        self.mname = mname
        self.scalar_op = scalar_op
        self.infer_value = infer_value
        self.cache = {}

    def normalize_args(self, args):
        """If infer_value is False, return broadened arguments."""
        if not self.infer_value:
            args = tuple(broaden(a, None) for a in args)
        return args

    def generate_graph(self, args):
        """Generate the graph."""
        sig = tuple(arg.values[SHAPE]
                    if isinstance(arg, AbstractArray) else False
                    for arg in args)
        if sig not in self.cache:
            g = Graph()
            g.flags['core'] = True
            g.debug.name = self.mname
            shapes = [x for x in sig if x is not False]

            is_array_op = len(shapes) > 0
            params = []
            for i, arg in enumerate(args):
                p = g.add_parameter()
                p.debug.name = f'x{i + 1}'
                if is_array_op and not isinstance(arg, AbstractArray):
                    p = g.apply(to_array, p)
                params.append(p)

            if is_array_op:
                try:
                    final_shape = reduce(broadcast_shape, shapes)
                except ValueError as e:
                    raise MyiaShapeError(e.args[0])
                if any(dim is ANYTHING for dim in final_shape):
                    # We will need to get the shapes dynamically
                    def _build(a, b):
                        return g.apply(broadcast_shape, a, b)
                    argshapes = [g.apply(shape, p) for p in params]
                    final_shape = reduce(_build, argshapes)

            transformed = []
            for arg, p in zip(args, params):
                if is_array_op:
                    sh = arg.values.get(SHAPE, ())
                    if final_shape != sh:
                        p = g.apply(distribute, p, final_shape)
                transformed.append(p)

            if is_array_op:
                fn = self.scalar_op or self
                g.output = g.apply(array_map, fn, *transformed)
            else:
                first, *rest = transformed
                fn = g.apply(P.getattr, first, self.mname)
                g.output = g.apply(fn, *rest)
            self.cache[sig] = g
        return self.cache[sig]

    def __call__(self, *args):
        """Python version of Elemwise's functionality."""
        return array_map(py_registry[self.scalar_op], *args)


add = Elemwise('__add__', P.scalar_add, name='add')
sub = Elemwise('__sub__', P.scalar_sub, name='sub')
mul = Elemwise('__mul__', P.scalar_mul, name='mul')
truediv = Elemwise('__truediv__', name='truediv')
floordiv = Elemwise('__floordiv__', name='floordiv')
mod = Elemwise('__mod__', P.scalar_mod, name='mod')
pow = Elemwise('__pow__', P.scalar_pow, name='pow')


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
tanh = MultitypeGraph('tanh')


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


@tanh.register(Number)
@core
def _tanh(x):
    return scalar_tanh(x)


eq = Elemwise('__eq__', P.scalar_eq, infer_value=True, name='eq')
lt = Elemwise('__lt__', P.scalar_lt, infer_value=True, name='lt')
gt = Elemwise('__gt__', P.scalar_gt, infer_value=True, name='gt')
ne = Elemwise('__ne__', P.scalar_ne, infer_value=True, name='ne')
le = Elemwise('__le__', P.scalar_le, infer_value=True, name='le')
ge = Elemwise('__ge__', P.scalar_ge, infer_value=True, name='ge')


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
        if (hastype(x, i8) or hastype(x, u8) or
                hastype(x, i16) or hastype(x, u16)):
            return scalar_div(scalar_cast(x, f32), scalar_cast(y, f32))
        return scalar_div(scalar_cast(x, f64), scalar_cast(y, f64))
    else:
        # This branch is only here to trigger a type check error.
        return scalar_div(x, y)


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

    idx: int
    seq: object

    @core(ignore_values=True)
    def __myia_hasnext__(self):
        """Whether the index is past the length of the sequence."""
        return self.idx < len(self.seq)

    @core(ignore_values=True)
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

    def generate_graph(self, args):
        """Generate tail specialized for the given Tuple type.

        tail(x) generates make_tuple(x[1], x[2], ...)
        """
        if len(args) != 1:
            raise MyiaTypeError('tail takes one argument')
        a, = args
        if not isinstance(a, AbstractTuple):
            raise MyiaTypeError('tail requires a Tuple')
        if len(a.elements) == 0:
            raise MyiaTypeError('tail requires a non-empty Tuple')
        g = Graph()
        g.flags['core'] = True
        tup = g.add_parameter()
        tup.debug.name = "tup"
        elems = [g.apply(P.tuple_getitem, tup, i)
                 for i in range(1, len(a.elements))]
        g.output = g.apply(P.make_tuple, *elems)
        return g


tail = Tail('tail')


#################
# Array methods #
#################

@core
def sum(x):
    """Implementation of `sum`."""
    return array_reduce(scalar_add, x, ())


@core
def to_array(x):
    """Implementation of `to_array`."""
    return x.__myia_to_array__()


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


@tanh.register(Array)
@core
def array_tanh(xs):
    """Implementation of `array_tanh`."""
    return array_map(scalar_tanh, xs)


_leaf_add = MultitypeGraph('hyper_add')


@_leaf_add.register(Number, Number)
@core
def _scalar_add(x, y):
    return scalar_add(x, y)


@_leaf_add.register(EnvType, EnvType)
@core
def _sm_add(x, y):
    return env_add(x, y)


hyper_add = HyperMap(name='hyper_add', fn_leaf=_leaf_add)


_leaf_zeros_like = MultitypeGraph('zeros_like')


@_leaf_zeros_like.register(AbstractFunction)
@core
def _function_zero(_):
    return newenv


@_leaf_zeros_like.register(AbstractError)
@core
def _dead_zero(x):
    return x


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
    scalar_zero = scalar_cast(0, typeof(xs).element)
    return distribute(to_array(scalar_zero), shape(xs))


zeros_like = HyperMap(
    name='zeros_like',
    nonleaf=(AbstractTuple, AbstractList, AbstractClass),
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

    def __init__(self, fn_rec=None, loop_mask=None):
        """Initialize the list_map.

        Arguments:
            fn_rec: The function to map over, or None if the function to
                map over must be provided as the first argument.
            loop_mask: Either None or a list of booleans such that the
                argument i is a list to iterate on if loop_mask[i] is
                True, and otherwise it is not iterated on.

        """
        self.fn_rec = fn_rec
        self.loop_mask = loop_mask
        super().__init__(self._decorate_name('list_map'))

    def _decorate_name(self, name):
        if self.fn_rec is None:
            return name
        else:
            return f'{name}[{self.fn_rec}]'

    def generate_graph(self, args):
        """Return a graph for the number of lists."""
        mask = self.loop_mask
        nfn = 1 if self.fn_rec is None else 0
        if mask is None:
            mask = (True,) * (len(args) - nfn)
        nmin = 1 + nfn
        if len(args) < nmin:
            raise MyiaTypeError(f'{self} takes at least {nmin} arguments')
        for t, m in zip(args[nfn:], mask):
            if m and not isinstance(t, AbstractList):
                raise MyiaTypeError(f'list_map requires lists, not {t}')

        g = Graph()
        g.flags['core'] = True
        g.flags['ignore_values'] = True
        g.debug.name = self._decorate_name('list_map')
        fn = self.fn_rec or g.add_parameter()
        gargs = [g.add_parameter() for _ in args[nfn:]]

        values = [g.apply(P.list_getitem, a, 0) if m else a
                  for a, m in zip(gargs, mask)]
        resl = g.apply(P.make_list, g.apply(fn, *values))

        gnext = Graph()
        gnext.debug.name = self._decorate_name('lm_body')
        gnext.flags['ignore_values'] = True
        gcond = Graph()
        gcond.debug.name = self._decorate_name('lm_cond')
        gcond.flags['ignore_values'] = True

        def make_cond(g):
            fn = self.fn_rec or g.add_parameter()
            curri = g.add_parameter()
            resl = g.add_parameter()
            gargs2 = [g.add_parameter() for _ in gargs]
            lists2 = [a for a, m in zip(gargs2, mask) if m]
            hasnexts = [g.apply(P.scalar_lt, curri, g.apply(P.list_len, l))
                        for l in lists2]
            cond = reduce(lambda a, b: g.apply(P.bool_and, a, b), hasnexts)
            gtrue = Graph()
            gtrue.debug.name = self._decorate_name('lm_ftrue')
            gtrue.flags['core'] = True
            gtrue.flags['ignore_values'] = True
            if self.fn_rec is None:
                gtrue.output = gtrue.apply(gnext, fn, curri, resl, *gargs2)
            else:
                gtrue.output = gtrue.apply(gnext, curri, resl, *gargs2)
            gfalse = Graph()
            gfalse.debug.name = self._decorate_name('lm_ffalse')
            gfalse.flags['core'] = True
            gfalse.flags['ignore_values'] = True
            gfalse.output = resl
            g.output = g.apply(g.apply(P.switch, cond, gtrue, gfalse))

        def make_next(g):
            fn = self.fn_rec or g.add_parameter()
            curri = g.add_parameter()
            resl = g.add_parameter()
            gargs2 = [g.add_parameter() for _ in gargs]
            values = [g.apply(P.list_getitem, a, curri) if m else a
                      for a, m in zip(gargs2, mask)]
            resl = g.apply(P.list_append, resl, g.apply(fn, *values))
            nexti = g.apply(P.scalar_add, curri, 1)
            if self.fn_rec is None:
                g.output = g.apply(gcond, fn, nexti, resl, *gargs2)
            else:
                g.output = g.apply(gcond, nexti, resl, *gargs2)

        make_cond(gcond)
        make_next(gnext)
        if self.fn_rec is None:
            g.output = g.apply(gcond, fn, 1, resl, *gargs)
        else:
            g.output = g.apply(gcond, 1, resl, *gargs)

        g.flags['forbid_inlining'] = True
        return g

    def __call__(self, fn, *lists):
        """Python implementation of list_map."""
        assert self.fn_rec is None
        from .prim.py_implementations import list_map
        return list_map(fn, *lists)


list_map = ListMap()


@core
def _cast_helper(x, model):
    t = typeof(model)
    if hastype(model, Array):
        return scalar_to_array(scalar_cast(x, t.element))
    else:
        return scalar_cast(x, t)


class GradOperation(MetaGraph):
    """Implements the grad(f) operation.

    grad(f)(x, ...) returns df(x, ...)/dx. Derivatives of other inputs are
    thrown out.

    TODO: This currently will not work on primitives, but it is an easy fix.
    We just need to know how many parameters f takes.
    """

    def __init__(self,
                 name,
                 return_value=False):
        """Initialize GradOperation."""
        super().__init__(name)
        self.return_value = return_value

    def make_gf(self, jf, orig_params, dbg, wrt,
                sens_param=False, apply_j=False):
        """Make the graph for the grad.

        If wrt is an integer, the wrt-th gradient will be returned directly.
        If it is a tuple of integers, then a tuple of the specified gradients
        will be returned in the same order (duplicates are allowed).

        If self.return_value is True, a tuple will always be returned and the
        first element will be the return value of the function. The other
        elements will be the gradients.
        """
        with About(dbg, 'grad'):
            df = Graph()
            df.flags['core'] = True

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

        if isinstance(wrt, int):
            direct_return = True
            wrt = [wrt]
        else:
            direct_return = False

        bapp = df.apply(bprop, bprop_arg)
        elems = []
        if self.return_value:
            elems.append(out)
        for idx in wrt:
            elems.append(df.apply(P.tuple_getitem, bapp, idx + 1))

        if len(elems) == 1 and direct_return:
            df.output = elems[0]
        else:
            df.output = df.apply(P.make_tuple, *elems)
        return df

    def generate_graph(self, args):
        """Generate the graph."""
        ft, *raw_wrt = args
        assert isinstance(ft, AbstractFunction)
        gf = ft.get_unique()
        assert isinstance(gf, GraphFunction)
        g = gf.graph

        wrt = []
        for entry in raw_wrt:
            value = build_value(entry, default=None)
            if value is None:
                raise MyiaTypeError(
                    'grad must statically know which parameters to use'
                )
            elif value == '*':
                wrt += range(len(g.parameters))
            elif isinstance(value, str):
                parameter_names = [short_labeler.name(p)
                                   for p in g.parameters]
                try:
                    wrt.append(parameter_names.index(value))
                except ValueError:
                    raise MyiaTypeError(f'No parameter named "{value}""')
            elif isinstance(value, int):
                wrt.append(value)
            else:
                raise MyiaTypeError(
                    'The parameters to use for grad must be strings or'
                    ' integers.'
                )

        if len(wrt) == 1:
            wrt = wrt[0]
        elif len(wrt) == 0:
            wrt = 0

        dfbuilder = Graph()
        dfbuilder.flags['core'] = True
        dfbuilder.debug.name = f"grad{len(g.parameters)}"

        with About(g.debug, 'copy'):
            fn = dfbuilder.add_parameter()
            for arg in args[1:]:
                # These are dummy parameters
                dfbuilder.add_parameter()

        with About(g.debug, 'grad_fprop'):
            jf = dfbuilder.apply(P.J, fn)

        df = self.make_gf(jf, g.parameters, g.debug, wrt=wrt)

        dfbuilder.output = Constant(df)

        return dfbuilder


grad = GradOperation('grad')
value_and_grad = GradOperation('value_and_grad', return_value=True)


class ArithmeticData:
    """Mixin to implement access to arithmetic operators.

    When used for a dataclass D, operations like D + D will add together
    all matching fields from the added instances.
    """

    __array_priority__ = 1_000_000

    @core
    def __add__(self, x):
        return hyper_map(add, self, x)

    @core
    def __sub__(self, x):
        return hyper_map(sub, self, x)

    @core
    def __mul__(self, x):
        return hyper_map(mul, self, x)

    @core
    def __truediv__(self, x):
        return hyper_map(truediv, self, x)

    @core
    def __floordiv__(self, x):
        return hyper_map(floordiv, self, x)

    @core
    def __mod__(self, x):
        return hyper_map(mod, self, x)

    @core
    def __pow__(self, x):
        return hyper_map(pow, self, x)

    @core
    def __pos__(self):
        return hyper_map(uadd, self)

    @core
    def __neg__(self):
        return hyper_map(usub, self)

    @core
    def __radd__(self, x):
        return hyper_map(add, x, self)

    @core
    def __rsub__(self, x):
        return hyper_map(sub, x, self)

    @core
    def __rmul__(self, x):
        return hyper_map(mul, x, self)

    @core
    def __rtruediv__(self, x):
        return hyper_map(truediv, x, self)

    @core
    def __rfloordiv__(self, x):
        return hyper_map(floordiv, x, self)

    @core
    def __rmod__(self, x):
        return hyper_map(mod, x, self)

    @core
    def __rpow__(self, x):
        return hyper_map(pow, x, self)
