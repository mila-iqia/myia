"""Definitions for the primitive `scalar_le`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_le(x: Number, y: Number) -> Bool:
    """Implement `scalar_le`."""
    assert_scalar(x, y)
    return x <= y


infer_scalar_le = UniformPrimitiveInferrer.partial(
    prim=P.scalar_le, impl=pyimpl_scalar_le, infer_value=True
)


@bprop_to_grad_transform(P.scalar_le)
def bprop_scalar_le(x, y, out, dout):
    """Backpropagator for `scalar_le`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    "name": "scalar_le",
    "registered_name": "scalar_le",
    "mapping": P.scalar_le,
    "python_implementation": pyimpl_scalar_le,
}


__primitive_defaults__ = {
    "name": "scalar_le",
    "registered_name": "scalar_le",
    "type": "backend",
    "python_implementation": pyimpl_scalar_le,
    "inferrer_constructor": infer_scalar_le,
    "grad_transform": bprop_scalar_le,
}
