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


# NOTE: The following operations should be implemented as dunder methods,
# but the lack of __rop__ support means we have to use elemwise for now.
# This should be fixed in the future.

add = elemwise('add', '__add__', operations.scalar_add, pyop=operator.add)
sub = elemwise('sub', '__sub__', operations.scalar_sub, pyop=operator.sub)
mul = elemwise('mul', '__mul__', operations.scalar_mul, pyop=operator.mul)
mod = elemwise('mod', '__mod__', operations.scalar_mod, pyop=operator.mod)
pow = elemwise('pow', '__pow__', operations.scalar_pow, pyop=operator.pow)

truediv = elemwise('truediv', '__truediv__', None, pyop=operator.truediv)
floordiv = elemwise('floordiv', '__floordiv__', None, pyop=operator.floordiv)

eq = elemwise('eq', '__eq__', operations.scalar_eq,
              infer_value=True, pyop=operator.eq)
lt = elemwise('lt', '__lt__', operations.scalar_lt,
              infer_value=True, pyop=operator.lt)
gt = elemwise('gt', '__gt__', operations.scalar_gt,
              infer_value=True, pyop=operator.gt)
ne = elemwise('ne', '__ne__', operations.scalar_ne,
              infer_value=True, pyop=operator.ne)
le = elemwise('le', '__le__', operations.scalar_le,
              infer_value=True, pyop=operator.le)
ge = elemwise('ge', '__ge__', operations.scalar_ge,
              infer_value=True, pyop=operator.ge)

pos = dunder_method_protocol('pos')
neg = dunder_method_protocol('neg')
floor = dunder_method_protocol('floor')
trunc = dunder_method_protocol('trunc')
matmul = dunder_method_protocol_2('matmul')
bool = dunder_method_protocol('bool')
and_ = dunder_method_protocol_2('and')
or_ = dunder_method_protocol_2('or')
len = dunder_method_protocol('len')
getitem = dunder_method_protocol_2('getitem')
myia_iter = dunder_method_protocol('myia_iter')
myia_next = dunder_method_protocol('myia_next')
myia_hasnext = dunder_method_protocol('myia_hasnext')
myia_to_array = dunder_method_protocol_2('myia_to_array')
