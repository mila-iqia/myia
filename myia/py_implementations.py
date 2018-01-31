"""Implementations for the debug VM."""


from typing import Dict, Callable
from copy import copy
from myia import primops


implementations: Dict[primops.Primitive, Callable] = {}


def make_registrar(store):
    """Create a decorator to associate a function to a primitive.

    The primitive will be associated to the function in the given
    `store`.
    """
    def deco1(prim):
        """Take the primitive."""
        def deco2(fn):
            """Decorate the function."""
            store[prim] = fn
            return fn
        return deco2
    return deco1


register = make_registrar(implementations)


@register(primops.add)
def add(x, y):
    """Implement `add`."""
    return x + y


@register(primops.sub)
def sub(x, y):
    """Implement `sub`."""
    return x - y


@register(primops.mul)
def mul(x, y):
    """Implement `mul`."""
    return x * y


@register(primops.div)
def div(x, y):
    """Implement `div`."""
    return x / y


@register(primops.mod)
def mod(x, y):
    """Implement `mod`."""
    return x % y


@register(primops.pow)
def pow(x, y):
    """Implement `pow`."""
    return x ** y


@register(primops.uadd)
def uadd(x):
    """Implement `iadd`."""
    return x


@register(primops.usub)
def usub(x):
    """Implement `isub`."""
    return -x


@register(primops.eq)
def eq(x, y):
    """Implement `eq`."""
    return x == y


@register(primops.lt)
def lt(x, y):
    """Implement `lt`."""
    return x < y


@register(primops.gt)
def gt(x, y):
    """Implement `gt`."""
    return x > y


@register(primops.ne)
def ne(x, y):
    """Implement `ne`."""
    return x != y


@register(primops.le)
def le(x, y):
    """Implement `le`."""
    return x <= y


@register(primops.ge)
def ge(x, y):
    """Implement `ge`."""
    return x >= y


@register(primops.not_)
def not_(x):
    """Implement `not_`."""
    return not x


@register(primops.tuple)
def tuple(*elems):
    """Implement `tuple`."""
    return elems


@register(primops.getitem)
def getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@register(primops.setitem)
def setitem(data, item, value):
    """Implement `setitem`."""
    data2 = copy(data)
    data2[item] = value
    return data2


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x
