"""Implementations for the debug VM."""


from typing import Dict, Callable
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
def impl_index(tup, idx):
    """Implement `index`."""
    return tup[idx]


@impl
def impl_return_(x):
    """Implement `return_`."""
    return x
