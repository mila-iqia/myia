"""Definitions for the primitive `scalar_to_array`."""

import numpy as np

from ..lib import (
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractScalar,
    AbstractType,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import array_to_scalar
from . import primitives as P


def pyimpl_scalar_to_array(x, t):
    """Implement `scalar_to_array`."""
    return np.array(x)


@standard_prim(P.scalar_to_array)
async def infer_scalar_to_array(
    self, engine, a: AbstractScalar, t: AbstractType
):
    """Infer the return type of primitive `scalar_to_array`."""
    tp = t.element
    assert isinstance(tp, AbstractArray)
    return AbstractArray(a, {SHAPE: (), TYPE: tp.xtype()})


@bprop_to_grad_transform(P.scalar_to_array)
def bprop_scalar_to_array(x, t, out, dout):
    """Backpropagator for primitive `scalar_to_array`."""
    return (array_to_scalar(dout), t)


__operation_defaults__ = {
    "name": "scalar_to_array",
    "registered_name": "scalar_to_array",
    "mapping": P.scalar_to_array,
    "python_implementation": pyimpl_scalar_to_array,
}


__primitive_defaults__ = {
    "name": "scalar_to_array",
    "registered_name": "scalar_to_array",
    "type": "backend",
    "python_implementation": pyimpl_scalar_to_array,
    "inferrer_constructor": infer_scalar_to_array,
    "grad_transform": bprop_scalar_to_array,
}
