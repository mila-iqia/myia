"""Definitions for the primitive `array_setitem`."""

from copy import copy

from .. import lib
from ..lib import standard_prim
from . import primitives as P


def pyimpl_array_setitem(data, begin, end, strides, value):
    """Implement `list/array_setitem`."""
    idx = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
    data2 = copy(data)
    data2[idx] = value
    return data2


@standard_prim(P.array_setitem)
async def infer_array_setitem(self, engine,
                              a: lib.AbstractArray,
                              begin: lib.u64tup_typecheck,
                              end: lib.u64tup_typecheck,
                              strides: lib.i64tup_typecheck,
                              value: lib.AbstractArray):
    """Infer the return type of primitive `array_setitem`."""
    return a


__operation_defaults__ = {
    'name': 'array_setitem',
    'registered_name': 'array_setitem',
    'mapping': P.array_setitem,
    'python_implementation': pyimpl_array_setitem,
}


__primitive_defaults__ = {
    'name': 'array_setitem',
    'registered_name': 'array_setitem',
    'type': 'backend',
    'python_implementation': pyimpl_array_setitem,
    'inferrer_constructor': infer_array_setitem,
    'grad_transform': None,
}
