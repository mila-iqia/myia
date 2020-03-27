"""Definitions for the primitive `distribute`."""

import numpy as np

from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    MyiaShapeError,
    bprop_to_grad_transform,
    force_pending,
    standard_prim,
    u64tup_typecheck,
)
from ..operations import array_reduce, scalar_add, shape, zeros_like
from . import primitives as P


def pyimpl_distribute(v, shape):
    """Implement `distribute`."""
    return np.broadcast_to(v, shape)


@standard_prim(P.distribute)
async def infer_distribute(
    self, engine, a: AbstractArray, _shp: u64tup_typecheck
):
    """Infer the return type of primitive `distribute`."""
    shp = tuple(x.xvalue() for x in _shp.elements)
    a_shp = await force_pending(a.xshape())
    delta = len(shp) - len(a_shp)
    if delta < 0:
        raise MyiaShapeError("Cannot distribute to smaller shape")
    elif delta > 0:
        a_shp = (1,) * delta + a_shp
    for vs, s in zip(a_shp, shp):
        if vs != s and vs not in (1, ANYTHING) and s not in (1, ANYTHING):
            raise MyiaShapeError("Cannot change shape when distributing")
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@bprop_to_grad_transform(P.distribute)
def bprop_distribute(arr, shp, out, dout):
    """Backpropagator for primitive `distribute`."""
    return (array_reduce(scalar_add, dout, shape(arr)), zeros_like(shp))


__operation_defaults__ = {
    "name": "distribute",
    "registered_name": "distribute",
    "mapping": P.distribute,
    "python_implementation": pyimpl_distribute,
}


__primitive_defaults__ = {
    "name": "distribute",
    "registered_name": "distribute",
    "type": "backend",
    "python_implementation": pyimpl_distribute,
    "inferrer_constructor": infer_distribute,
    "grad_transform": bprop_distribute,
}
