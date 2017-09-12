
from typing import List, Any, Union
from .main import symbol_associator, impl_bank
from ..stx import Symbol
from ..interpret import \
    PrimitiveImpl, FunctionImpl, ClosureImpl, evaluate
from ..symbols import ZERO
from ..grad import JX
from ..inference.types import typeof
from ..lib import StructuralMap


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


##############################################
# Implementations of myia's global functions #
##############################################


@impl_interp_smap
def interp_smap_add(x, y):
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


@impl_interp
def interp_J(x: Any) -> Any:
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
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(interp_J(a) for a in x)
    elif isinstance(x, list):
        return list(interp_J(a) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return JX(x, 0)
    elif isinstance(x, ClosureImpl):
        c = ClosureImpl(JX(x.fn, len(x.args)),
                        interp_J(tuple(x.args)))
        return c
    elif x is None or x is ZERO:
        return x
    else:
        raise TypeError(f'Invalid argument for J: {x}')


@impl_interp
def interp_Jinv(x: Any) -> Any:
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
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(interp_Jinv(a) for a in x)
    elif isinstance(x, list):
        return list(interp_Jinv(a) for a in x)
    elif isinstance(x, PrimitiveImpl):
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
    elif isinstance(x, ClosureImpl):
        c = ClosureImpl(interp_Jinv(x.fn),
                        interp_Jinv(tuple(x.args)))
        return c
    elif x is None or x is ZERO:
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
    # TODO: rename to zeros_like
    return interp_fill(x, 0)


@impl_interp
def interp_ones_like(x):
    return interp_fill(x, 1)


@impl_interp
def interp_mapadd(x: Any, y: Any) -> Any:
    """
    Element-wise addition.

    >>> mapadd(10, 9)
    19
    >>> mapadd((1, 2, (3, 4)), (4, 3, (2, 1)))
    (5, 5, (5, 5))

    As a special case, ``mapadd(ZERO, x) == x``

    Implements the "⊕" (circled plus) operator in Pearlmutter & Siskind.
    """
    # TODO: this should be add, but add concatenates tuples, whereas
    # this adds their values element-wise.
    if y is ZERO:
        raise TypeError(f'ZERO should never be found as the '
                        'second argument to mapadd.')
    elif x is ZERO:
        return y
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x + y
    elif type(x) is not type(y):
        raise TypeError(f'Cannot mapadd {x} and {y} (not same type).')
    elif isinstance(x, tuple):
        assert len(x) == len(y)
        return tuple(interp_mapadd(a, b) for a, b in zip(x, y))
    elif isinstance(x, list):
        assert len(x) == len(y)
        return list(interp_mapadd(a, b) for a, b in zip(x, y))
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot mapadd values of type {type(x)}')


@impl_interp
def interp_assert_true(x, msg):
    assert x, msg


@impl_interp
def interp_type(x):
    return typeof(x)


@impl_interp
def interp_shape(x):
    return x.shape
