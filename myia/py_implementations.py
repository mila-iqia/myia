"""Implementations for the debug VM."""


from typing import Callable
from copy import copy
from myia import primops
from myia.utils import Registry


implementations: Registry[primops.Primitive, Callable] = Registry()
register = implementations.register


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


@register(primops.make_tuple)
def make_tuple(*elems):
    """Implement `make_tuple`."""
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


py_getattr = getattr
py_setattr = setattr


@register(primops.getattr)
def getattr(data, attr):
    """Implement `getattr`."""
    return py_getattr(data, attr)


@register(primops.setattr)
def setattr(data, attr, value):
    """Implement `setattr`."""
    data2 = copy(data)
    py_setattr(data2, attr, value)
    return data2


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x
