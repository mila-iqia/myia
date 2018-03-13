"""Implementations for the debug VM."""

from typing import Callable, Dict
from copy import copy
from .signatures import SIGNATURES
from myia.dtype import Bool, Int, Float, Tuple, Type

from ..utils import Registry

from . import ops as primops

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


@register(primops.cons_tuple)
def cons_tuple(head, tail):
    """Implement `cons_tuple`."""
    return (head,) + tail


@register(primops.head)
def head(tup):
    """Implement `head`."""
    return tup[0]


@register(primops.tail)
def tail(tup):
    """Implement `tail`."""
    return tup[1:]


@register(primops.getitem)
def getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@register(primops.setitem)
def setitem(data, item, value):
    """Implement `setitem`."""
    if isinstance(data, tuple):
        return tuple(value if i == item else x
                     for i, x in enumerate(data))
    else:
        data2 = copy(data)
        data2[item] = value
        return data2


py_getattr = getattr  # type: ignore
py_setattr = setattr  # type: ignore


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


TYPE_MAP: Dict[type, Type] = {
    bool: Bool(),
    float: Float(64),
    int: Int(64),
}


@register(primops.typeof)
def typeof(value) -> Type:
    """Return the Type for a python constant.

    This doesn't support deeply nested or recursive values.
    """
    tt = type(value)
    if tt in TYPE_MAP:
        return TYPE_MAP[tt]
    if tt == tuple:
        etts = tuple(typeof(e) for e in value)
        return Tuple(etts)
    if tt is primops.Primitive:
        return SIGNATURES[value]
    raise TypeError(f"Cannot assign type to: {value}")


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x
