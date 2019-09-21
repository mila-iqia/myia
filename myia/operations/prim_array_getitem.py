"""Definitions for the primitive `array_getitem`."""

import operator

from .. import lib
from ..lib import SHAPE, TYPE, bprop_to_grad_transform, standard_prim
from ..operations import array_setitem, zeros_like
from . import primitives as P


def pyimpl_array_getitem(data, begin, end, strides):
    """Implement `getitem`."""
    idx = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
    return data[idx]


def _ceildiv(x, y):
    return -(-x // y)


@standard_prim(P.array_getitem)
async def infer_array_getitem(self, engine,
                              a: lib.AbstractArray,
                              begin: lib.u64tup_typecheck,
                              end: lib.u64tup_typecheck,
                              strides: lib.i64tup_typecheck):
    """Infer the return type of primitive `array_getitem`."""
    begin = tuple(self.require_constant(e, argnum=f'"1:begin[{edx}]"')
                  for edx, e in enumerate(begin.elements))
    end = tuple(self.require_constant(e, argnum=f'"2:end[{edx}]"')
                for edx, e in enumerate(end.elements))
    strides = tuple(self.require_constant(e, argnum=f'"3:strides[{edx}]"')
                    for edx, e in enumerate(strides.elements))

    shp_before_stride = map(operator.sub, end, begin)
    shp = tuple(map(_ceildiv, shp_before_stride, map(abs, strides)))

    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@bprop_to_grad_transform(P.array_getitem, ignore_values=False)
def bprop_array_getitem(data, begin, end, strides, out, dout):
    """Backpropagator for primitive `array_getitem`."""
    return (array_setitem(zeros_like(data), begin, end, strides, dout),
            zeros_like(begin), zeros_like(end), zeros_like(strides))


__operation_defaults__ = {
    'name': 'array_getitem',
    'registered_name': 'array_getitem',
    'mapping': P.array_getitem,
    'python_implementation': pyimpl_array_getitem,
}


__primitive_defaults__ = {
    'name': 'array_getitem',
    'registered_name': 'array_getitem',
    'type': 'backend',
    'python_implementation': pyimpl_array_getitem,
    'inferrer_constructor': infer_array_getitem,
    'grad_transform': bprop_array_getitem,
}
