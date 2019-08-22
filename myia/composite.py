"""Implementations of primitives as graphs."""


import operator
from dataclasses import dataclass
from functools import reduce

from . import operations
from .abstract import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractClassBase,
    AbstractDict,
    AbstractError,
    AbstractFunction,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    broaden,
    build_value,
)
from .dtype import (
    Array,
    Bool,
    EnvType,
    Nil,
    Number,
    f32,
    f64,
    i8,
    i16,
    u8,
    u16,
)
from .hypermap import HyperMap, hyper_map
from .ir import Graph, MetaGraph, MultitypeGraph
from .prim import ops as P
from .prim.py_implementations import (
    array_map,
    array_reduce,
    bool_eq,
    bool_not,
    broadcast_shape,
    distribute,
    env_add,
    hastype,
    py_registry,
    scalar_add,
    scalar_cast,
    scalar_cos,
    scalar_div,
    scalar_exp,
    scalar_log,
    scalar_sin,
    scalar_tan,
    scalar_tanh,
    shape,
    tuple_getitem,
    typeof,
)
from .utils import MyiaShapeError, MyiaTypeError, Slice, check_nargs, newenv


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
        'reference': True,
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

    def normalize_args_sync(self, args):
        """If infer_value is False, return broadened arguments."""
        if not self.infer_value:
            args = tuple(broaden(a) for a in args)
        return args

    def make_signature(self, args):
        """Create the signature: whether arguments are arrays, and shapes."""
        return tuple((type(arg), arg.values[SHAPE])
                     if isinstance(arg, AbstractArray) else (None, False)
                     for arg in args)

    def generate_graph(self, sig):
        """Generate the graph."""
        g = Graph()
        g.set_flags('core', 'reference')
        g.debug.name = self.mname
        shapes = [x for _, x in sig if x is not False]
        is_array_op = len(shapes) > 0
        if is_array_op:
            array_types = [t for t, _ in sig if t is not None]
            array_type = array_types[0](ANYTHING, {SHAPE: ANYTHING})
        params = []
        for i, (t, _) in enumerate(sig):
            p = g.add_parameter()
            p.debug.name = f'x{i + 1}'
            if is_array_op and t is None:
                p = g.apply(to_array, p, array_type)
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
        for (_, sh), p in zip(sig, params):
            if is_array_op:
                sh = sh or ()
                if final_shape != sh:
                    p = g.apply(distribute, p, final_shape)
            transformed.append(p)

        if is_array_op:
            fn = self.scalar_op or self
            g.output = g.apply(array_map, fn, *transformed)
        else:
            first, *rest = transformed
            fn = g.apply(operations.getattr, first, self.mname)
            g.output = g.apply(fn, *rest)
        return g

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
        raise Exception("Incompatible types for division.")


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
def array_iter(xs):
    """Iterator for Array."""
    return SequenceIterator(0, xs)


@core
def tuple_next(xs):
    """Next tuple."""
    return xs[0], xs[1:]


@core
def tuple_hasnext(xs):
    """Whether the tuple is empty or not."""
    return len(xs) > 0


class TupleReorganizer(MetaGraph):
    """Parametrizable MetaGraph to combine or extract tuples."""

    def __init__(self, name, gen):
        """Initialize a TupleReorganizer."""
        super().__init__(name)
        self.gen = gen

    def map_tuples(self, g, params, tups):
        """Map each element of each tuple to a getitem on the parameter."""
        rval = []
        for tup, param in zip(tups, params):
            if not isinstance(tup, AbstractTuple):
                raise MyiaTypeError(f'Expected AbstractTuple, not {tup}')
            rval.append([
                g.apply(P.tuple_getitem, param, i)
                for i, elem in enumerate(tup.elements)
            ])
        return rval

    def generate_graph(self, args):
        """Generate the graph."""
        g = Graph()
        g.debug.name = self.gen.__name__
        for arg in args:
            g.add_parameter()
        g.output = self.gen(self, g, args)
        return g


def tuple_reorganizer(fn):
    """Shortcut to create a new TupleReorganizer from a function."""
    return TupleReorganizer(name=fn.__name__, gen=fn)


@tuple_reorganizer
def tuple_concat(self, g, args):
    """Metagraph for tuple concatenation."""
    tups = self.map_tuples(g, g.parameters, args)
    return g.apply(P.make_tuple, *reduce(operator.add, tups))


@tuple_reorganizer
def tuple_getslice(self, g, args):
    """Metagraph for getting a slice from a tuple."""
    tuparg, start, stop, step = check_nargs('tail', 4, args)
    try:
        start = build_value(start)
        stop = build_value(stop)
        step = build_value(step)
    except ValueError:
        raise MyiaTypeError('Slice start, stop and step must be static')
    tup, = self.map_tuples(g, g.parameters[:1], [tuparg])
    return g.apply(P.make_tuple, *tup[start:stop:step])


@core
def tuple_get(t, item):
    """Implementation of `tuple.__getitem__`."""
    if hastype(item, Slice):
        return tuple_getslice(t, item.start, item.stop, item.step)
    else:
        return tuple_getitem(t, item)


#################
# Array methods #
#################

@core
def sum(x):
    """Implementation of `sum`."""
    return array_reduce(scalar_add, x, ())


@core
def to_array(x, t):
    """Implementation of `to_array`."""
    return x.__myia_to_array__(t)


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


@core
def nil_eq(a, b):
    """Implementation of `equal` (only use with Nil types)."""
    if hastype(a, Nil) and hastype(b, Nil):
        return True
    else:
        return False


@core
def nil_ne(a, b):
    """Implementation of `not_equal` (only use with Nil types)."""
    return not nil_eq(a, b)


@core
def nil_bool(x):
    """Converting Nil (None) to Bool returns False."""
    return False


_leaf_add = MultitypeGraph('gadd')


@_leaf_add.register(Number, Number)
@core
def _scalar_add(x, y):
    return scalar_add(x, y)


@_leaf_add.register(EnvType, EnvType)
@core
def _sm_add(x, y):
    return env_add(x, y)


@_leaf_add.register(Nil, Nil)
@core
def _nil_add(x, y):
    return None


gadd = HyperMap(name='gadd', fn_leaf=_leaf_add,
                broadcast=False, trust_union_match=True)


_leaf_zeros_like = MultitypeGraph('zeros_like')


@_leaf_zeros_like.register(AbstractFunction(value=ANYTHING))
@core
def _function_zero(_):
    return newenv


@_leaf_zeros_like.register(AbstractError(ANYTHING))
@core
def _dead_zero(x):
    return x


@_leaf_zeros_like.register(Bool)
@core
def _bool_zero(_):
    return False


@_leaf_zeros_like.register(Nil)
@core
def _nil_zero(_):
    return None


@_leaf_zeros_like.register(Number)
@core
def _scalar_zero(x):
    return scalar_cast(0, typeof(x))


@_leaf_zeros_like.register(Array)
@core
def _array_zero(xs):
    scalar_zero = scalar_cast(0, typeof(xs).element)
    return distribute(to_array(scalar_zero, typeof(xs)), shape(xs))


zeros_like = HyperMap(
    name='zeros_like',
    nonleaf=(AbstractTuple, AbstractClassBase,
             AbstractUnion, AbstractTaggedUnion, AbstractDict),
    fn_leaf=_leaf_zeros_like
)


class IsCompare(MetaGraph):
    """Implementation of Is Compare (i.e. 'is' and 'is not')."""

    def __init__(self, do_not=False):
        """Initialize the is_compare.

        Arguments:
            do_not: If True, this graph becomes a negative ("is not")
                comparison. If False, this graph remains a psotive ("is")
                comparison.

        """
        super().__init__('IsCompare')
        self.do_not = do_not

    def normalize_args_sync(self, args):
        """Return broadened arguments."""
        return tuple(broaden(a) for a in args)

    def generate_graph(self, args):
        """Make the graph for the IsCompare.

        This requires that least one argument of
        comparison be a Bool or None.

        Generate Boolean Compare if False and
        Equal_To (and not Equal_To) compare if True.
        """
        assert len(args) == 2
        a, b = args
        g = Graph()
        g.debug.name = "is_not" if self.do_not else "is_"
        g.set_flags('core', 'reference')
        pa = g.add_parameter()
        pb = g.add_parameter()

        valid_types = (Bool, Nil)
        if ((not isinstance(a, AbstractScalar))
            or (a.values[TYPE] not in valid_types)) \
                and ((not isinstance(b, AbstractScalar))
                     or (b.values[TYPE] not in valid_types)):
            if not self.do_not:
                raise MyiaTypeError(
                    f'The operator "is" must have at ' +
                    f'least one argument be a Bool or None'
                )
            else:
                raise MyiaTypeError(
                    f'The operator "is not" must have at ' +
                    f'least one argument be a Bool or None'
                )

        if a != b:
            g.return_ = g.apply(P.return_, self.do_not)
        else:
            cmp_fn = g.apply(operations.getattr, pa,
                             "__ne__" if self.do_not else "__eq__")
            g.output = g.apply(cmp_fn, pb)
        return g


is_ = IsCompare()
is_not = IsCompare(do_not=True)


@core
def list_reduce(fn, lst, dftl):
    """Implementation of list_reduce."""
    res = dftl
    for elem in lst:
        res = fn(res, elem)
    return res


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


@dataclass
class Range:  # pragma: no cover
    """Implement a Range in Myia."""
    start: object
    stop: object
    step: object

    def __myia_iter__(self):
        return self

    def __myia_next__(self):
        return self.start, Range(self.start + self.step, self.stop, self.step)

    def __myia_hasnext__(self):
        return self.start < self.stop


@core
def range_(start, stop=None, step=None):
    """Myia implementation of the standard range function."""
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    return Range(start, stop, step)


@dataclass
class Zip2:
    """Implement zip with two arguments."""
    iter1: object
    iter2: object

    def __len__(self):
        return len(self.iter1)

    def __myia_iter__(self):
        return self

    def __myia_next__(self):
        nxt1, iter1 = next(self.iter1)
        nxt2, iter2 = next(self.iter2)
        return (nxt1, nxt2), Zip2(iter1, iter2)

    def __myia_hasnext__(self):
        return hasnext(self.iter1) and hasnext(self.iter2)


@core
def zip_(seq1, seq2):
    """Myia implementation of the standard zip function."""
    return Zip2(iter(seq1), iter(seq2))


@core
def enumerate_(seq):
    """Myia implementation of the standard enumerate function."""
    return zip_(range(len(seq)), seq)
