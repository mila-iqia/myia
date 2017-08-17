
from typing import List, Any, Union
from .runtime import \
    impl, PrimitiveImpl, FunctionImpl, ClosureImpl
from ..util import Keyword
from ..symbols import ZERO


##############################################
# Implementations of myia's global functions #
##############################################


@impl
def impl_add(x, y):
    return x + y


@impl
def impl_subtract(x, y):
    return x - y


@impl
def impl_multiply(x, y):
    return x * y


@impl
def impl_divide(x, y):
    return x / y


@impl
def impl_unary_subtract(x):
    return -x


@impl
def impl_equal(x, y):
    return x == y


@impl
def impl_less(x, y):
    return x < y


@impl
def impl_greater(x, y):
    return x > y


@impl
def impl_len(t):
    return len(t)


@impl
def impl_range(t):
    return tuple(range(t))


@impl
def impl_index(t, i):
    return t[i]


@impl
def impl_first(t):
    return t[0]


@impl
def impl_second(t):
    return t[1]


@impl
def impl_getattr(obj, attr):
    return getattr(obj, attr)


@impl
def impl_map(f, xs):
    return tuple(map(f, xs))


@impl
def impl_reduce(f, xs):
    return reduce(f, xs)


@impl
def impl_enumerate(xs):
    return tuple(enumerate(xs))


@impl
def impl_switch(cond, t, f):
    if cond:
        return t
    else:
        return f

@impl
def impl_identity(x):
    return x


################################################
# Implementation of primitives needed for Grad #
################################################


@impl
def impl_fill(x: Any, value: Union[int, float]) -> Any:
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
        return tuple(fill(a, value) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return ()
    elif isinstance(x, ClosureImpl):
        return tuple(fill(a, value) for a in x.args)
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot create a {value} conformant with {x}')


@impl
def impl_zeros_like(x):
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
    return fill(x, 0)


@impl
def impl_ones_like(x):
    return fill(x, 1)


@impl
def impl_mapadd(x: Any, y: Any) -> Any:
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
        return tuple(impl_mapadd(a, b) for a, b in zip(x, y))
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot mapadd values of type {type(x)}')


fill = impl_fill
zeros_like = impl_zeros_like
ones_like = impl_ones_like


__all__: List[Any] = []
