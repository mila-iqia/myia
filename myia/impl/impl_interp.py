
from typing import List, Any, Union
from copy import copy
import numpy
from .main import symbol_associator, impl_bank
from ..stx import Symbol
from ..lib import ZERO
from ..transform import find_grad
from ..inference.types import typeof
from ..lib import \
    Primitive, Closure, Function, \
    StructuralMap, default_structural_map_dispatch
from ..symbols import object_map
from ..parse import parse_function
from ..util.debug import Breakpoint, BreakpointMode


_ = True


pylen = len
pylist = list
pyrange = range
myiaClosure = Closure
pytuple = tuple
pygetattr = getattr
pysetattr = setattr
pytype = type
pymap = map
pyenumerate = enumerate
pyException = Exception
pyprint = print
pyslice = slice


@symbol_associator('')
def impl_interp(sym, name, fn):
    """
    Define the implementation for the given symbol.
    The implementation will be set in ``root_globals``.
    """
    prim = Primitive(fn, name=sym)
    impl_bank['interp'][sym] = prim
    object_map[prim] = sym
    return prim


@symbol_associator('')
def impl_interp_myia(sym, name, fn):
    """
    Define the implementation for the given symbol using
    Myia. The implementation will be set in ``root_globals``.
    """
    lbda = parse_function(fn)
    impl_bank['interp'][sym] = lbda
    prim = lbda  # eenvs.run_env(lbda)
    # object_map[prim] = lbda #sym
    return prim


def impl_interp_smap(dispatch):
    @symbol_associator('')
    def deco(sym, name, fn):
        mfn = StructuralMap(fn, dispatch)
        prim = Primitive(mfn, name=sym)
        impl_bank['interp'][sym] = prim
        object_map[prim] = sym
        return prim

    if not isinstance(dispatch, dict):
        fn = dispatch
        dispatch = default_structural_map_dispatch
        return deco(fn)
    else:
        dispatch = {**dispatch, **default_structural_map_dispatch}
        return deco


##############################################
# Implementations of myia's global functions #
##############################################


@impl_interp_smap
def add(x, y):
    """
    Element-wise addition. Note that unlike Python's addition, this
    adds tuples element-by-element. ``add`` also works on Records and
    Closures. Addition of a primitive to itself, ``P + P``, is legal and
    yields ``P``.

    >>> add(10, 9)
    19
    >>> add((1, 2, (3, 4)), (4, 3, (2, 1)))
    (5, 5, (5, 5))

    As a special case, ``add(ZERO, x) == x``. The converse, ``add(x, ZERO)``,
    is considered an error.
    """
    return x + y


@impl_interp_smap
def subtract(x, y):
    return x - y


@impl_interp_smap
def multiply(x, y):
    return x * y


@impl_interp_smap
def divide(x, y):
    return x / y


@impl_interp
def dot(x, y):
    return x @ y


@impl_interp
def transpose(x):
    return x.T


@impl_interp_smap
def unary_subtract(x):
    return -x


@impl_interp_smap
def power(x, y):
    return x ** y


@impl_interp_smap
def exp(x):
    return numpy.exp(x)


@impl_interp_smap
def log(x):
    return numpy.log(x)


@impl_interp
def sum(xs):
    return numpy.sum(xs)


@impl_interp
def equal(x, y):
    return x == y


@impl_interp
def less(x, y):
    return x < y


@impl_interp
def greater(x, y):
    return x > y


@impl_interp
def broadcast(arrs):
    return tuple(numpy.broadcast_arrays(*arrs))


@impl_interp
def fit(arr, shp):
    if isinstance(arr, (int, float)):
        arr = numpy.asarray(arr)
    orig_shp = shp
    shp0 = arr.shape
    n0 = len(shp0)
    n = len(shp)
    if n0 > n:
        shp = ((1,) * (n0 - n)) + shp
    if n > n0:
        shp0 = ((1,) * (n - n0)) + shp0
    arr = arr.reshape(shp0)
    sum_axes = tuple(i for i, (s0, s1) in enumerate(list(zip(shp0, shp)))
                     if s1 == 1 and s0 != 1)
    arr = arr.sum(axis=sum_axes, keepdims=True)
    arr, _ = numpy.broadcast_arrays(arr, numpy.zeros(shp))
    arr = arr.reshape(orig_shp)
    return arr


@impl_interp
def len(t):
    return pylen(t)


@impl_interp
def range(t):
    return pylist(pyrange(t))


@impl_interp
def Closure(fn, args):
    return myiaClosure(fn, args)


@impl_interp
def closure_fn(clos):
    return clos.fn


@impl_interp
def closure_args(clos):
    return clos.args


@impl_interp
def partial(fn, *args):
    return myiaClosure(fn, args)


@impl_interp
def mktuple(*args):
    return pytuple(args)


@impl_interp
def mklist(*args):
    return pylist(args)


@impl_interp
def index(t, i):
    return t[i]


@impl_interp
def setslice(t, i, v):
    typ = pytype(t)
    return typ(v if i == j else orig
               for j, orig in enumerate(t))


@impl_interp
def first(t):
    return t[0]


@impl_interp
def second(t):
    return t[1]


@impl_interp
def getattr(obj, attr):
    return pygetattr(obj, attr)


@impl_interp
def setattr(obj, attr, value):
    if hasattr(obj, '__variant__'):
        return obj.__variant__(attr, value)
    else:
        obj2 = copy(obj)
        pysetattr(obj2, attr, value)
        return obj2


@impl_interp
def map(f, xs):
    return pytype(xs)(pymap(f, xs))


@impl_interp
def reduce(f, xs):
    v = xs[0]
    for x in xs[1:]:
        v = f(v, x)
    return v


@impl_interp
def enumerate(xs):
    return pytype(xs)(pyenumerate(xs))


@impl_interp
def switch(cond, t, f):
    if cond:
        return t
    else:
        return f


@impl_interp
def if_(cond, t, f):
    if cond:
        return t()
    else:
        return f()


@impl_interp
def identity(x):
    return x


@impl_interp
def raise_exception(x):
    # TODO: wrap the exception, and have the interpreter catch it
    # and display the Myia stack trace instead of the Python
    # interpreter's own stack trace.
    raise x


@impl_interp
def Exception(x):
    return pyException(x)


@impl_interp
def print(x):
    pyprint(x)


################################################
# Implementation of primitives needed for Grad #
################################################


def J_dispatch_closure(smap, clos):
    """
    Implements a special case for J on closures -- the transform
    of clos.fn depends on clos.args, whereas the default behavior for
    StructuralMap considers them separately.
    """
    c = myiaClosure(J_fn(clos.fn, pylen(clos.args)),
                    smap(pytuple(clos.args)))
    return c


def J_fn(x, nargs_closure):
    """
    Helper function for the gradient of Primitive or
    Function, given nargs_closure closure arguments.

    See previous section on Partial Application for the
    purpose of the ``nargs_closure`` argument.
    """
    if isinstance(x, Primitive):
        ref = x.name
    elif isinstance(x, Function):
        ref = x.ast.ref
    else:
        raise TypeError(f'J_fn applied on wrong type: {x}')
    return x.universe[find_grad(ref, nargs_closure)]


@impl_interp_smap({myiaClosure: J_dispatch_closure})
def J(x):
    """
    Return a Grad-transformed version of this data.

    * On scalars, this is the identity function.
    * On a data structure, this applies ``J`` on each element and
      returns a data structure with the same shape.
    * On a function of type ``T -> U``, this returns the
      Grad-transformed function, with signature (more or less)
      ``J(T) -> (J(U), S(U) -> S(T))``. That is to say, it returns
      J-transformed outputs and a backpropagator function that
      takes an output sentisitivity and returns an input
      sensitivity (don't look for an S type operator, I made that
      up (J(T) isn't exactly correct either, since it's not a type
      operator), but for what it's worth, ``zeros_like(x)`` would
      have the signature ``T -> S(T)``).

    Implements the J operator in Pearlmutter & Siskind.
    """
    if isinstance(x, (Primitive, Function)):
        return J_fn(x, 0)
    elif isinstance(x, (int, float, bool, str)) or x is None or x is ZERO:
        return x
    elif isinstance(x, (numpy.ndarray, numpy.generic)):
        return x
    else:
        raise TypeError(f'Invalid argument for J: {x}')


@impl_interp_smap
def Jinv(x):
    """
    Undo the effect of ``J``.

    * On scalars, this is the identity function.
    * On a data structure, this applies ``Jinv`` on each element and
      returns a data structure with the same shape.
    * On a function, this undoes the effect of ``J``. This should
      *never* be applied on a function that was not the result of
      transforming through ``J``.

    Implements the J^{-1} operator in Pearlmutter & Siskind.
    """
    if isinstance(x, Primitive):
        raise Exception('Primitives have no primals.')
    elif isinstance(x, Function):
        assert isinstance(x.primal_sym, Symbol)
        primal = x.universe[x.primal_sym]
        if not isinstance(primal, (Function, Primitive)):
            raise Exception('Should be Function, but found:'
                            f' {primal}, type {type(primal)},'
                            f' for {x.primal_sym}')
        return primal
    elif isinstance(x, (int, float, bool, str)) or x is None or x is ZERO:
        return x
    elif isinstance(x, (numpy.ndarray, numpy.generic)):
        return x
    else:
        raise TypeError(f'Invalid argument for Jinv: {x} ({pytype(x)})')


@impl_interp_myia
def grad1(fn):
    jfn = J(fn)

    def g(x):
        _, bprop = jfn(J(x))
        return bprop(1.0)[1]
    return g


@impl_interp_myia
def grad2(fn):
    jfn = J(fn)

    def g(x, y):
        _, bprop = jfn(J(x), J(y))
        return bprop(1.0)[1]
    return g


@impl_interp_myia
def grad3(fn):
    jfn = J(fn)

    def g(x, y, z):
        _, bprop = jfn(J(x), J(y), J(z))
        return bprop(1.0)[1]
    return g


def zeros_like_closure(smap, x):
    return smap(x.args)


@impl_interp_smap({myiaClosure: zeros_like_closure})
def zeros_like(x):
    """
    Creates a structure just like ``x`` but "zeroed out."

    If ``x`` is a Primitive or a Function, this returns
    (). If ``x`` is a Closure, this returns a Closure on
    a zeroed environment.

    >>> zeros_like(17)
    0
    >>> zeros_like((1, 2, (3, 4)))
    (0, 0, (0, 0))
    >>> x = 10; zeros_like(lambda y: x + y)  # (metaphorically)
    lambda y: 0 + y
    """
    if isinstance(x, (int, float, bool, numpy.int64)):
        return 0
    elif isinstance(x, (Primitive, Function)):
        return x
    elif x is None or x is ZERO:
        return x
    else:
        raise TypeError(f'Cannot create a zero conformant with {x}')


@impl_interp
def assert_true(x, msg):
    assert x, msg


@impl_interp
def type(x):
    return typeof(x)


@impl_interp
def shape(x):
    return x.shape


@impl_interp
def concat(*xs):
    assert all(isinstance(x, list) for x in xs) or \
        all(isinstance(x, tuple) for x in xs)
    return sum(xs, pytype(xs[0])())


@impl_interp
def slice(start, stop=None, step=None):
    return pyslice(start, stop, step)


@impl_interp
def breakpoint(mode=BreakpointMode.FORWARD):
    return Breakpoint(mode)
