
from typing import List, Any
from .runtime import impl


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

__all__: List[Any] = []
