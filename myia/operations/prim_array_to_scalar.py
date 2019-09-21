"""Definitions for the primitive `array_to_scalar`."""

import numpy as np

from ..lib import (
    AbstractArray,
    MyiaShapeError,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import scalar_to_array, typeof
from . import primitives as P


def pyimpl_array_to_scalar(x):
    """Implement `array_to_scalar`."""
    assert isinstance(x, np.ndarray)
    return x.item()


@standard_prim(P.array_to_scalar)
async def infer_array_to_scalar(self, engine, a: AbstractArray):
    """Infer the return type of primitive `array_to_scalar`."""
    a_shp = a.xshape()
    if len(a_shp) != 0:
        raise MyiaShapeError("array_to_scalar requires shape ()")
    return a.element


@bprop_to_grad_transform(P.array_to_scalar)
def bprop_array_to_scalar(x, out, dout):
    """Backpropagator for primitive `array_to_scalar`."""
    return (scalar_to_array(dout, typeof(x)),)


__operation_defaults__ = {
    'name': 'array_to_scalar',
    'registered_name': 'array_to_scalar',
    'mapping': P.array_to_scalar,
    'python_implementation': pyimpl_array_to_scalar,
}


__primitive_defaults__ = {
    'name': 'array_to_scalar',
    'registered_name': 'array_to_scalar',
    'type': 'backend',
    'python_implementation': pyimpl_array_to_scalar,
    'inferrer_constructor': infer_array_to_scalar,
    'grad_transform': bprop_array_to_scalar,
}
