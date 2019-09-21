"""Definitions for the primitive `dot`."""

import numpy as np

from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    MyiaShapeError,
    MyiaTypeError,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import dot, transpose
from . import primitives as P


def pyimpl_dot(a, b):
    """Implement `dot`."""
    return np.dot(a, b)


@standard_prim(P.dot)
async def infer_dot(self, engine, a: AbstractArray, b: AbstractArray):
    """Infer the return type of primitive `dot`."""
    a_shp = a.xshape()
    b_shp = b.xshape()
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in dot: {a_shp} and {b_shp}"
        )
    engine.abstract_merge(a.element, b.element)
    c_shp = (a_shp[0], b_shp[1])

    if a.xtype() != b.xtype():
        raise MyiaTypeError(
            f'Expect array of type {a.xtype()} '
            f'to have same type as array of type {b.xtype()}')

    return type(a)(a.element, {SHAPE: c_shp, TYPE: a.xtype()})


@bprop_to_grad_transform(P.dot)
def bprop_dot(x, y, out, dout):
    """Backpropagator for primitive `dot`."""
    return (dot(dout, transpose(y, (1, 0))),
            dot(transpose(x, (1, 0)), dout))


__operation_defaults__ = {
    'name': 'dot',
    'registered_name': 'dot',
    'mapping': P.dot,
    'python_implementation': pyimpl_dot,
}


__primitive_defaults__ = {
    'name': 'dot',
    'registered_name': 'dot',
    'type': 'backend',
    'python_implementation': pyimpl_dot,
    'inferrer_constructor': infer_dot,
    'grad_transform': bprop_dot,
}
