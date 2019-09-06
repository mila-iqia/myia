"""Implementations of primitives as graphs."""


import operator
from dataclasses import dataclass
from functools import reduce

from . import operations
from .abstract import (
    ANYTHING,
    AbstractArray,
    AbstractBottom,
    AbstractClassBase,
    AbstractDict,
    AbstractError,
    AbstractFunction,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    build_value,
    myia_static,
)
from .hypermap import HyperMap, hyper_map
from .ir import Graph, MetaGraph, MultitypeGraph
from .prim import ops as P, py_implementations as py
from .prim.py_implementations import (
    array_map,
    array_reduce,
    bool_eq,
    bool_not,
    distribute,
    env_add,
    hastype,
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
    string_eq,
    tuple_getitem,
    typeof,
)
from .utils import MyiaTypeError, Slice, check_nargs, core, newenv
from .xtype import Bool, EnvType, Nil, Number, f32, f64, i8, i16, u8, u16


class Elemwise(HyperMap):
    """Generate a graph for an elemwise operation.

    * If any argument is an array:
      * All scalar arguments are converted to arrays using scalar_to_array.
      * The arguments are all broadcasted and array_map is called on them.
    * Otherwise, we return getattr(arg1, mname)(arg2, ...)
    """

    def __init__(self, mname, scalar_op=None, infer_value=False, name=None):
        """Initialize an Elemwise."""
        super().__init__(
            fn_leaf=scalar_op or self,
            infer_value=infer_value,
            name=name,
            broadcast=True,
            nonleaf=(AbstractArray,)
        )
        self.mname = mname

    def make_signature(self, args):
        """This erases type information that isn't useful.

        The only information kept is whether an arg is an array or not, and its
        shape/type/etc. (not its dtype). Other information is thrown away
        because it is not relevant to graph generation.
        """
        def chg(arg):
            if isinstance(arg, AbstractArray):
                return AbstractArray(
                    AbstractBottom(),
                    arg.values
                )
            else:
                return AbstractBottom()
        return tuple(chg(arg) for arg in args)

    def make_leaf(self, g, fnarg, argmap):
        """Generate a call at leaf level.

        This does not use self.fn_leaf.
        """
        assert fnarg is None
        if self.mname is None:
            return super().make_leaf(g, fnarg, argmap)
        else:
            first, *rest = argmap.keys()
            fn = g.apply(operations.getattr, first, self.mname)
            return g.apply(fn, *rest)


def dunder_method_protocol(name):
    """Define a function that calls a certain method (unary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data):
        return getattr(data, attr)()

    return protocol


def dunder_method_protocol_2(name):
    """Define a function that calls a certain method (binary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data, x):
        return getattr(data, attr)(x)

    return protocol


def dunder_method_protocol_3(name):
    """Define a function that calls a certain method (ternary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data, x, y):
        return getattr(data, attr)(x, y)

    return protocol


def arrayify(name, op):
    """Create an array_map over the op."""
    @core(name=name)
    def arrayified(xs):
        return array_map(op, xs)

    return arrayified


#########################
# Arithmetic operations #
#########################


add = Elemwise('__add__', py.scalar_add, name='add')
sub = Elemwise('__sub__', py.scalar_sub, name='sub')
mul = Elemwise('__mul__', py.scalar_mul, name='mul')
truediv = Elemwise('__truediv__', name='truediv')
floordiv = Elemwise('__floordiv__', name='floordiv')
mod = Elemwise('__mod__', py.scalar_mod, name='mod')
pow = Elemwise('__pow__', py.scalar_pow, name='pow')
exp = Elemwise(None, scalar_exp, name='exp')
log = Elemwise(None, scalar_log, name='log')
sin = Elemwise(None, scalar_sin, name='sin')
cos = Elemwise(None, scalar_cos, name='cos')
tan = Elemwise(None, scalar_tan, name='tan')
tanh = Elemwise(None, scalar_tanh, name='tanh')
uadd = dunder_method_protocol('pos')
usub = dunder_method_protocol('neg')
floor = dunder_method_protocol('floor')
trunc = dunder_method_protocol('trunc')
scalar_truediv = dunder_method_protocol_2('truediv')
scalar_floordiv = dunder_method_protocol_2('floordiv')
matmul = dunder_method_protocol_2('matmul')


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
def float_floordiv(x, y):
    """Implementation of `float_floordiv`."""
    return floor(x / y)


###############
# Comparisons #
###############


@core
def not_(x):
    """Implementation of `not`."""
    return bool_not(bool(x))


@core
def is_not(x, y):
    """Implementation of the `is not` operator."""
    return not (x is y)


@core
def bool_ne(x, y):
    """Implementation of `bool_ne`."""
    return bool_not(bool_eq(x, y))


@core
def string_ne(x, y):
    """Implementation of `string_ne`."""
    return bool_not(string_eq(x, y))


eq = Elemwise('__eq__', py.scalar_eq, infer_value=True, name='eq')
lt = Elemwise('__lt__', py.scalar_lt, infer_value=True, name='lt')
gt = Elemwise('__gt__', py.scalar_gt, infer_value=True, name='gt')
ne = Elemwise('__ne__', py.scalar_ne, infer_value=True, name='ne')
le = Elemwise('__le__', py.scalar_le, infer_value=True, name='le')
ge = Elemwise('__ge__', py.scalar_ge, infer_value=True, name='ge')
bool = dunder_method_protocol('bool')
and_ = dunder_method_protocol_2('and')
or_ = dunder_method_protocol_2('or')


@core
def nil_eq(a, b):
    """Implementation of `equal` (only use with Nil types)."""
    return a is None and b is None


@core
def nil_ne(a, b):
    """Implementation of `not_equal` (only use with Nil types)."""
    return not nil_eq(a, b)


@core
def nil_bool(x):
    """Converting Nil (None) to Bool returns False."""
    return False


@core
def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


@core
def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0


#############
# Iteration #
#############


_len = dunder_method_protocol('len')
getitem = dunder_method_protocol_2('getitem')
setitem = dunder_method_protocol_3('setitem')
iter = dunder_method_protocol('myia_iter')
next = dunder_method_protocol('myia_next')
hasnext = dunder_method_protocol('myia_hasnext')


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


to_array = dunder_method_protocol_2('myia_to_array')
array_floor = arrayify('array_floor', floor)
array_trunc = arrayify('array_trunc', trunc)
array_uadd = arrayify('array_uadd', uadd)
array_usub = arrayify('array_usub', usub)


@core
def sum(x):
    """Implementation of `sum`."""
    return array_reduce(scalar_add, x, ())


@core
def ndim(arr):
    """Return the number of dimensions of an array."""
    return len(shape(arr))


@myia_static
def _revperm(n):
    return tuple(reversed(range(n)))


@core
def transpose(arr, permutation=None):
    """Transpose an array."""
    if permutation is None:
        permutation = _revperm(ndim(arr))
    return P.transpose(arr, permutation)


########
# gadd #
########


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


##############
# zeros_like #
##############


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


@_leaf_zeros_like.register(AbstractArray)
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


##################
# ArithmeticData #
##################


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


#################
# Miscellaneous #
#################


@core
def list_reduce(fn, lst, dftl):
    """Implementation of list_reduce."""
    res = dftl
    for elem in lst:
        res = fn(res, elem)
    return res


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
class Zip2:  # pragma: no cover
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
