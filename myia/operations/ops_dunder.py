"""Define operations that work by calling a __method__."""

import operator

from .. import operations
from ..lib import core
from .ops_array import elemwise


def _operation(name, fn, pyop):
    regname = name
    if regname in ['and', 'or']:
        regname += '_'
    return {
        'name': name,
        'registered_name': regname,
        'mapping': fn,
        'python_implementation': pyop,
    }


def dunder_method_protocol(name, pyop=None):
    """Define a function that calls a certain method (unary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data):
        return getattr(data, attr)()

    return _operation(name, protocol, pyop)


def dunder_method_protocol_2(name, pyop=None):
    """Define a function that calls a certain method (binary)."""
    attr = f'__{name}__'

    @core(name=name)
    def protocol(data, x):
        return getattr(data, attr)(x)

    return _operation(name, protocol, pyop)


def exc_fallback(x, y):
    raise Exception('Not implemented')


def is_fallback(x, y):
    return x is y


def is_not_fallback(x, y):
    return x is not y


def dunder_protocol_general(name, lattr, rattr, infer_value=False,
                            fallback=exc_fallback, pyop=None):
    """Define a function that calls a certain method (binary)."""
    @core(name=name)
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


# # NOTE: The following operations should be implemented as dunder methods,
# # but the lack of __rop__ support means we have to use elemwise for now.
# # This should be fixed in the future.

# add = elemwise('add', '__add__', operations.scalar_add, pyop=operator.add)
# sub = elemwise('sub', '__sub__', operations.scalar_sub, pyop=operator.sub)
# mul = elemwise('mul', '__mul__', operations.scalar_mul, pyop=operator.mul)
# mod = elemwise('mod', '__mod__', operations.scalar_mod, pyop=operator.mod)
# pow = elemwise('pow', '__pow__', operations.scalar_pow, pyop=operator.pow)

# truediv = elemwise('truediv', '__truediv__', None, pyop=operator.truediv)
# floordiv = elemwise('floordiv', '__floordiv__', None, pyop=operator.floordiv)

# eq = elemwise('eq', '__eq__', operations.scalar_eq,
#               infer_value=True, pyop=operator.eq)
# lt = elemwise('lt', '__lt__', operations.scalar_lt,
#               infer_value=True, pyop=operator.lt)
# gt = elemwise('gt', '__gt__', operations.scalar_gt,
#               infer_value=True, pyop=operator.gt)
# ne = elemwise('ne', '__ne__', operations.scalar_ne,
#               infer_value=True, pyop=operator.ne)
# le = elemwise('le', '__le__', operations.scalar_le,
#               infer_value=True, pyop=operator.le)
# ge = elemwise('ge', '__ge__', operations.scalar_ge,
#               infer_value=True, pyop=operator.ge)

# pos = dunder_method_protocol('pos')
# neg = dunder_method_protocol('neg')
# floor = dunder_method_protocol('floor')
# trunc = dunder_method_protocol('trunc')
# matmul = dunder_method_protocol_2('matmul')
# bool = dunder_method_protocol('bool')
# and_ = dunder_method_protocol_2('and')
# or_ = dunder_method_protocol_2('or')
# len = dunder_method_protocol('len')
# getitem = dunder_method_protocol_2('getitem')
# myia_iter = dunder_method_protocol('myia_iter')
# myia_next = dunder_method_protocol('myia_next')
# myia_hasnext = dunder_method_protocol('myia_hasnext')
# myia_to_array = dunder_method_protocol_2('myia_to_array')


add = dunder_protocol_general(name='add', lattr='__add__', rattr= '__radd__',
                              pyop=operator.add)
sub = dunder_protocol_general(name='sub', lattr='__sub__', rattr= '__rsub__',
                              pyop=operator.sub)
mul = dunder_protocol_general(name='mul', lattr='__mul__', rattr= '__rmul__',
                              pyop=operator.mul)
mod = dunder_protocol_general(name='mod', lattr='__mod__', rattr= '__rmod__',
                              pyop=operator.mod)
pow = dunder_protocol_general(name='pow', lattr='__pow__', rattr= '__rpow__',
                              pyop=operator.pow)

truediv = dunder_protocol_general(name='truediv', lattr='__truediv__',
                                  rattr='__truediv__',
                                  pyop=operator.truediv)
floordiv = dunder_protocol_general(name='floordiv', lattr='__floordiv__',
                                   rattr='__floordiv__',
                                   pyop=operator.floordiv)

eq = dunder_protocol_general(name='eq', lattr='__eq__', rattr='__eq__',
                             fallback=is_fallback,
                             infer_value=True, pyop=operator.eq)
lt = dunder_protocol_general(name='lt', lattr='__lt__', rattr='__ge__',
                             infer_value=True, pyop=operator.lt)
gt = dunder_protocol_general(name='gt', lattr='__gt__', rattr='__le__',
                             infer_value=True, pyop=operator.gt)
ne = dunder_protocol_general(name='ne', lattr='__ne__', rattr='__ne__',
                             fallback=is_not_fallback,
                             infer_value=True, pyop=operator.ne)
le = dunder_protocol_general(name='le', lattr='__le__', rattr='__gt__',
                             infer_value=True, pyop=operator.le)
ge = dunder_protocol_general(name='ge', lattr='__ge__', rattr='__lt__',
                             infer_value=True, pyop=operator.ge)

pos = dunder_method_protocol('pos')
neg = dunder_method_protocol('neg')
floor = dunder_method_protocol('floor')
trunc = dunder_method_protocol('trunc')
matmul = dunder_protocol_general(
    name='matmul', lattr='__matmul__', rattr='__rmatmul__',
    pyop=operator.matmul
)
bool = dunder_method_protocol('bool')
and_ = dunder_protocol_general(
    name='and', lattr='__and__', rattr='__rand__',
    pyop=operator.and_
)
or_ = dunder_protocol_general(
    name='or', lattr='__or__', rattr='__ror__',
    pyop=operator.or_
)
len = dunder_method_protocol('len')
getitem = dunder_method_protocol_2('getitem')
myia_iter = dunder_method_protocol('myia_iter')
myia_next = dunder_method_protocol('myia_next')
myia_hasnext = dunder_method_protocol('myia_hasnext')
myia_to_array = dunder_method_protocol_2('myia_to_array')
