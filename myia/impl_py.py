"""Implementations for the debug VM."""


from typing import Dict, Callable
from copy import copy
from myia import primops


implementations: Dict[primops.Primitive, Callable] = {}


def registrar(pfx, store):
    """Create a decorator to associate a function to a primitive.

    The function name will need to start with `pfx`, followed by
    the primitive's name in `primops`. It will then be associated
    to that Primitive in `store`.
    """
    def deco(fn):
        """Decorate the function."""
        name = fn.__name__
        assert name.startswith(pfx)
        name = name.split(pfx)[1]
        prim = getattr(primops, name)
        store[prim] = fn
        return fn
    return deco


impl = registrar('impl_', implementations)


@impl
def impl_add(x, y):
    """Implement `add`."""
    return x + y


@impl
def impl_sub(x, y):
    """Implement `sub`."""
    return x - y


@impl
def impl_mul(x, y):
    """Implement `mul`."""
    return x * y


@impl
def impl_div(x, y):
    """Implement `div`."""
    return x / y


@impl
def impl_mod(x, y):
    """Implement `mod`."""
    return x % y


@impl
def impl_pow(x, y):
    """Implement `pow`."""
    return x ** y


@impl
def impl_uadd(x):
    """Implement `iadd`."""
    return x


@impl
def impl_usub(x):
    """Implement `isub`."""
    return -x


@impl
def impl_eq(x, y):
    """Implement `eq`."""
    return x == y


@impl
def impl_lt(x, y):
    """Implement `lt`."""
    return x < y


@impl
def impl_gt(x, y):
    """Implement `gt`."""
    return x > y


@impl
def impl_ne(x, y):
    """Implement `ne`."""
    return x != y


@impl
def impl_le(x, y):
    """Implement `le`."""
    return x <= y


@impl
def impl_ge(x, y):
    """Implement `ge`."""
    return x >= y


@impl
def impl_not_(x):
    """Implement `not_`."""
    return not x


@impl
def impl_tuple(*elems):
    """Implement `tuple`."""
    return elems


@impl
def impl_getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@impl
def impl_setitem(data, item, value):
    """Implement `setitem`."""
    data2 = copy(data)
    data2[item] = value
    return data2


@impl
def impl_getattr(data, attr):
    """Implement `getattr`."""
    return getattr(data, attr)


@impl
def impl_setattr(data, attr, value):
    """Implement `setattr`."""
    data2 = copy(data)
    setattr(data2, attr, value)
    return data2


@impl
def impl_return_(x):
    """Implement `return_`."""
    return x
