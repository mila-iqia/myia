"""Definitions for the primitive `shape`."""

from .. import xtype
from ..lib import (
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
    AbstractTuple,
    bprop_to_grad_transform,
    force_pending,
    standard_prim,
)
from ..operations import zeros_like
from . import primitives as P


def pyimpl_shape(array):
    """Implement `shape`."""
    return array.shape


@standard_prim(P.shape)
async def infer_shape(self, engine, a: AbstractArray):
    """Infer the return type of primitive `shape`."""
    shp = await force_pending(a.xshape())
    values = [
        AbstractScalar({VALUE: entry, TYPE: xtype.UInt[64]}) for entry in shp
    ]
    return AbstractTuple(values)


@bprop_to_grad_transform(P.shape)
def bprop_shape(arr, out, dout):
    """Backpropagator for primitive `shape`."""
    return (zeros_like(arr),)


__operation_defaults__ = {
    "name": "shape",
    "registered_name": "shape",
    "mapping": P.shape,
    "python_implementation": pyimpl_shape,
}


__primitive_defaults__ = {
    "name": "shape",
    "registered_name": "shape",
    "type": "backend",
    "python_implementation": pyimpl_shape,
    "inferrer_constructor": infer_shape,
    "grad_transform": bprop_shape,
}
