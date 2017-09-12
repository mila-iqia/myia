
from typing import List, Any, Union
from .main import symbol_associator, impl_bank
from ..stx import Symbol
from ..interpret import \
    PrimitiveImpl, FunctionImpl, ClosureImpl, evaluate
from ..lib import ZERO
from ..grad import JX
from ..inference.types import typeof
from ..lib import StructuralMap, Closure, default_structural_map_dispatch


_ = True


@symbol_associator('interp')
def impl_interp(sym, name, fn):
    """
    Define the implementation for the given symbol.
    The implementation will be set in ``root_globals``
    and in the ``myia_builtins`` global.
    """
    prim = PrimitiveImpl(fn, name=sym)
    impl_bank['interp'][sym] = prim
    return prim


@symbol_associator('interp_smap')
def impl_interp_smap(sym, name, fn):
    mfn = StructuralMap(fn)
    prim = PrimitiveImpl(mfn, name=sym)
    impl_bank['interp'][sym] = prim
    return prim


def impl_interp_smap(dispatch):
    @symbol_associator('interp_smap')
    def deco(sym, name, fn):
        mfn = StructuralMap(fn, dispatch)
        prim = PrimitiveImpl(mfn, name=sym)
        impl_bank['interp'][sym] = prim
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
def interp_smap_add(x, y):
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
def interp_smap_subtract(x, y):
    return x - y


@impl_interp_smap
def interp_smap_multiply(x, y):
    return x * y


@impl_interp_smap
def interp_smap_divide(x, y):
    return x / y


@impl_interp
def interp_dot(x, y):
    return x @ y


@impl_interp_smap
def interp_smap_unary_subtract(x):
    return -x


@impl_interp
def interp_equal(x, y):
    return x == y


@impl_interp
def interp_less(x, y):
    return x < y


@impl_interp
def interp_greater(x, y):
    return x > y


@impl_interp
def interp_len(t):
    return len(t)


@impl_interp
def interp_range(t):
    return list(range(t))


@impl_interp
def interp_mktuple(*args):
    return tuple(args)


@impl_interp
def interp_index(t, i):
    return t[i]


@impl_interp
def interp_first(t):
    return t[0]


@impl_interp
def interp_second(t):
    return t[1]


@impl_interp
def interp_getattr(obj, attr):
    return getattr(obj, attr)


@impl_interp
def interp_map(f, xs):
    return tuple(map(f, xs))


@impl_interp
def interp_reduce(f, xs):
    return reduce(f, xs)


@impl_interp
def interp_enumerate(xs):
    return tuple(enumerate(xs))


@impl_interp
def interp_switch(cond, t, f):
    if cond:
        return t
    else:
        return f


@impl_interp
def interp_identity(x):
    return x


@impl_interp
def interp_raise_exception(x):
    # TODO: wrap the exception, and have the interpreter catch it
    # and display the Myia stack trace instead of the Python
    # interpreter's own stack trace.
    raise x


@impl_interp
def interp_Exception(x):
    return Exception(x)


@impl_interp
def interp_print(x):
    print(x)


################################################
# Implementation of primitives needed for Grad #
################################################


def J_dispatch_closure(smap, clos):
    """
    Implements a special case for J on closures -- the transform
    of clos.fn depends on clos.args, whereas the default behavior for
    StructuralMap considers them separately.
    """
    c = Closure(JX(clos.fn, len(clos.args)),
                smap(tuple(clos.args)))
    return c


@impl_interp_smap({Closure: J_dispatch_closure})
def interp_smap_J(x):
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
def interp_smap_Jinv(x):
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


@impl_interp
def interp_fill(x: Any, value: Union[int, float]) -> Any:
    """
    Creates a structure just like ``x`` but where each scalar element
    is set to ``value``.

    If ``x`` is a PrimitiveImpl or a FunctionImpl, this returns
    (). If ``x`` is a ClosureImpl, this returns a filled value
    for each value in the closure.
    """
    if isinstance(x, (int, float)):
        return value
    elif isinstance(x, tuple):
        return tuple(interp_fill(a, value) for a in x)
    elif isinstance(x, list):
        return list(interp_fill(a, value) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return ()
    elif isinstance(x, ClosureImpl):
        return tuple(interp_fill(a, value) for a in x.args)
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot create a {value} conformant with {x}')


@impl_interp
def interp_zeros_like(x):
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

    Implements the "0" operator in Pearlmutter & Siskind.
    """
    return interp_fill(x, 0)


@impl_interp
def interp_ones_like(x):
    return interp_fill(x, 1)


# @impl_interp_smap
# def interp_smap_mapadd(x, y):
#     return x + y


@impl_interp
def interp_assert_true(x, msg):
    assert x, msg


@impl_interp
def interp_type(x):
    return typeof(x)


@impl_interp
def interp_shape(x):
    return x.shape
