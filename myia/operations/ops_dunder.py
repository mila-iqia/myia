"""Define operations that work by calling a __method__."""

import operator

from ..lib import core
from .utils import OperationDefinition


def _operation(name, fn, pyop):
    regname = name
    if regname in ['and', 'or']:
        regname += '_'
    return OperationDefinition(
        name=name,
        registered_name=regname,
        mapping=fn,
        python_implementation=pyop,
    )


def dunder_protocol_unary(name, pyop=None):
    """Define a function that calls a certain method (unary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data):
        return getattr(data, attr)()

    return _operation(name, protocol, pyop)


def dunder_protocol_binary_simple(name, pyop=None):
    """Define a function that calls a certain method (binary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data, x):
        return getattr(data, attr)(x)

    return _operation(name, protocol, pyop)


@core(static_inline=True)
def exc_fallback(x, y):
    """Fallback for most dunder operations."""
    raise Exception('Not implemented')


@core(static_inline=True)
def is_fallback(x, y):
    """Fallback to x is y."""
    return x is y


@core(static_inline=True)
def is_not_fallback(x, y):
    """Fallback to x is not y."""
    return x is not y


def dunder_protocol_binary(name, lattr, rattr,
                           fallback=exc_fallback, pyop=None):
    """Define a function that calls a certain method (binary).

    This typically works as follows:
    * Try method lattr on the left operand.
    * If it doesn't exist or returns NotImplemented, try method rattr
      on the right operand.
    * If neither method exists, or they return NotImplemented, try the
      fallback (usually raise an exception).
    """
    @core(name=name, static_inline='inner')
    def protocol(x, y):
        rval = NotImplemented
        if hasattr(x, lattr):
            rval = getattr(x, lattr)(y)
        if rval is NotImplemented:
            if hasattr(y, rattr):
                rval = getattr(y, rattr)(x)
        if rval is NotImplemented:
            return fallback(x, y)
        return rval

    return _operation(name, protocol, pyop)


add = dunder_protocol_binary(
    name='add', lattr='__add__', rattr='__radd__',
    pyop=operator.add
)
sub = dunder_protocol_binary(
    name='sub', lattr='__sub__', rattr='__rsub__',
    pyop=operator.sub
)
mul = dunder_protocol_binary(
    name='mul', lattr='__mul__', rattr='__rmul__',
    pyop=operator.mul
)
mod = dunder_protocol_binary(
    name='mod', lattr='__mod__', rattr='__rmod__',
    pyop=operator.mod
)
pow = dunder_protocol_binary(
    name='pow', lattr='__pow__', rattr='__rpow__',
    pyop=operator.pow
)
truediv = dunder_protocol_binary(
    name='truediv', lattr='__truediv__', rattr='__truediv__',
    pyop=operator.truediv
)
floordiv = dunder_protocol_binary(
    name='floordiv', lattr='__floordiv__', rattr='__floordiv__',
    pyop=operator.floordiv
)
matmul = dunder_protocol_binary(
    name='matmul', lattr='__matmul__', rattr='__rmatmul__',
    pyop=operator.matmul
)
pos = dunder_protocol_unary('pos')
neg = dunder_protocol_unary('neg')
floor = dunder_protocol_unary('floor')
trunc = dunder_protocol_unary('trunc')

bool = dunder_protocol_unary('bool')
eq = dunder_protocol_binary(
    name='eq', lattr='__eq__', rattr='__eq__',
    pyop=operator.eq,
    fallback=is_fallback
)
lt = dunder_protocol_binary(
    name='lt', lattr='__lt__', rattr='__gt__',
    pyop=operator.lt
)
gt = dunder_protocol_binary(
    name='gt', lattr='__gt__', rattr='__lt__',
    pyop=operator.gt
)
ne = dunder_protocol_binary(
    name='ne', lattr='__ne__', rattr='__ne__',
    pyop=operator.ne,
    fallback=is_not_fallback
)
le = dunder_protocol_binary(
    name='le', lattr='__le__', rattr='__ge__',
    pyop=operator.le
)
ge = dunder_protocol_binary(
    name='ge', lattr='__ge__', rattr='__le__',
    pyop=operator.ge
)
and_ = dunder_protocol_binary(
    name='and', lattr='__and__', rattr='__rand__',
    pyop=operator.and_
)
or_ = dunder_protocol_binary(
    name='or', lattr='__or__', rattr='__ror__',
    pyop=operator.or_
)

getitem = dunder_protocol_binary_simple('getitem')
len = dunder_protocol_unary('len')
myia_iter = dunder_protocol_unary('myia_iter')
myia_next = dunder_protocol_unary('myia_next')
myia_hasnext = dunder_protocol_unary('myia_hasnext')
myia_to_array = dunder_protocol_binary_simple('myia_to_array')
