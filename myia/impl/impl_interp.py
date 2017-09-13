
from typing import List, Any, Union
from .main import symbol_associator, impl_bank
from ..stx import Symbol
from ..interpret import \
    PrimitiveImpl, FunctionImpl, ClosureImpl, evaluate
from ..lib import ZERO
from ..grad import JX
from ..inference.types import typeof
from ..lib import StructuralMap, Closure, default_structural_map_dispatch
from ..symbols import object_map


_ = True


pylen = len
pylist = list
pyrange = range
myiaClosure = Closure
pytuple = tuple
pygetattr = getattr
pytype = type
pymap = map
pyenumerate = enumerate
pyException = Exception
pyprint = print


@symbol_associator('')
def impl_interp(sym, name, fn):
    """
    Define the implementation for the given symbol.
    The implementation will be set in ``root_globals``
    and in the ``myia_builtins`` global.
    """
    prim = PrimitiveImpl(fn, name=sym)
    impl_bank['interp'][sym] = prim
    object_map[prim] = sym
    return prim


def impl_interp_smap(dispatch):
    @symbol_associator('')
    def deco(sym, name, fn):
        mfn = StructuralMap(fn, dispatch)
        prim = PrimitiveImpl(mfn, name=sym)
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


@impl_interp_smap
def unary_subtract(x):
    return -x


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
def mktuple(*args):
    return pytuple(args)


@impl_interp
def index(t, i):
    return t[i]


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
    c = myiaClosure(JX(clos.fn, pylen(clos.args)),
                    smap(pytuple(clos.args)))
    return c


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
    if isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return JX(x, 0)
    elif isinstance(x, (int, float, bool)) or x is None or x is ZERO:
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
    if isinstance(x, PrimitiveImpl):
        raise Exception('Primitives have no primals.')
    elif isinstance(x, FunctionImpl):
        assert x.primal_sym is not None
        if isinstance(x.primal_sym, Symbol):
            primal = evaluate(x.primal_sym, x.ast.global_env)
        else:
            primal = x.primal_sym
        if not isinstance(primal, (FunctionImpl, PrimitiveImpl)):
            raise Exception('Should be FunctionImpl, but found:'
                            f' {primal}, type {type(primal)},'
                            f' for {x.primal_sym}')
        return primal
    elif isinstance(x, (int, float, bool)) or x is None or x is ZERO:
        return x
    else:
        raise TypeError(f'Invalid argument for Jinv: {x}')


@impl_interp_smap
def zeros_like(x):
    """
    Creates a structure just like ``x`` but "zeroed out."

    If ``x`` is a PrimitiveImpl or a FunctionImpl, this returns
    (). If ``x`` is a ClosureImpl, this returns a zero
    for each value in the closure.

    >>> zeros_like(17)
    0
    >>> zeros_like((1, 2, (3, 4)))
    (0, 0, (0, 0))
    >>> zeros_like(lambda x, y: x + y)  # (metaphorically)
    ()
    >>> x = 10; zeros_like(lambda y: x + y)  # (metaphorically)
    (0,)
    """
    if isinstance(x, (int, float, bool)):
        return 0
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return x
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot create a {value} conformant with {x}')


@impl_interp
def assert_true(x, msg):
    assert x, msg


@impl_interp
def type(x):
    return typeof(x)


@impl_interp
def shape(x):
    return x.shape
