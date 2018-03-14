"""Implementations for the debug VM."""


import math
from typing import Callable
from types import FunctionType
from copy import copy
from . import ops as primops
from myia.anf_ir import Graph
from myia.utils import Registry, smap
from myia.vm import VMFrame


implementations: Registry[primops.Primitive, Callable] = Registry()
register = implementations.register


@register(primops.add)
def add(x, y):
    """Implement `add`."""
    if x is ZERO:
        return y
    elif y is ZERO:
        return x
    elif isinstance(x, tuple):
        # assert len(x) == len(y)
        l = max(len(x), len(y))
        x = Zero.pad(x, l)
        y = Zero.pad(y, l)
        return tuple(add(x2, y2) for x2, y2 in zip(x, y))
    else:
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


@register(primops.log)
def log(x):
    """Implement `log`."""
    return math.log(x)


@register(primops.exp)
def exp(x):
    """Implement `exp`."""
    return math.exp(x)


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
    elif data is ZERO:
        l = [ZERO for i in range(item + 1)]
        l[item] = value
        print('value is:', value)
        return tuple(l)
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


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x


@register(primops.J)
def J(x):
    from myia.grad_implementations import implementations
    from myia.anf_ir import Graph
    from myia.grad import grad

    if isinstance(x, primops.Primitive):
        return implementations[x]
    elif isinstance(x, Graph):
        return grad(x)
    elif isinstance(x, FunctionType):
        from myia.api import parse, compile
        g = parse(x)
        return compile(grad(g))
    elif isinstance(x, VMFrame.Closure):
        return VMFrame.Closure(J(x.graph), x.frame)
    elif isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return smap(J, x)
    elif x is ZERO:
        return ZERO
    else:
        raise TypeError(f'J is not defined on {type(x)}')


@register(primops.Jinv)
def Jinv(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return smap(Jinv, x)
    elif isinstance(x, Graph):
        assert x.primal
        return x.primal
    elif isinstance(x, VMFrame.Closure):
        return VMFrame.Closure(Jinv(x.graph), x.frame)
    elif x is ZERO:
        return ZERO
    else:
        raise TypeError(f'Jinv is not defined on {type(x)}')


class Zero:
    """Null object for addition.

    * ZERO + x is x
    * x + ZERO is x
    * ZERO[i] is ZERO
    """

    def __add__(self, z):
        return z

    def __radd__(self, z):
        return z

    def __getitem__(self, item):
        return self

    @staticmethod
    def pad(arr, n):
        m = len(arr)
        if m < n:
            return arr + type(arr)(ZERO for _ in range(n - m))
        else:
            return arr


ZERO = Zero()


@register(primops.zeros_like)
def zeros_like(x):
    def zero(x):
        if isinstance(x, VMFrame.Closure) or x is ZERO:
            return ZERO
        elif isinstance(x, (Graph, primops.Primitive)):
            return ()
        else:
            return type(x)(0)

    return smap(zero, x)
