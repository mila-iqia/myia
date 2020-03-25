"""Definitions for the primitive `scalar_ge`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_ge(x: Number, y: Number) -> Bool:
    """Implement `scalar_ge`."""
    assert_scalar(x, y)
    return x >= y


infer_scalar_ge = UniformPrimitiveInferrer.partial(
    prim=P.scalar_ge, impl=pyimpl_scalar_ge, infer_value=True
)


@bprop_to_grad_transform(P.scalar_ge)
def bprop_scalar_ge(x, y, out, dout):
    """Backpropagator for `scalar_ge`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    "name": "scalar_ge",
    "registered_name": "scalar_ge",
    "mapping": P.scalar_ge,
    "python_implementation": pyimpl_scalar_ge,
}


__primitive_defaults__ = {
    "name": "scalar_ge",
    "registered_name": "scalar_ge",
    "type": "backend",
    "python_implementation": pyimpl_scalar_ge,
    "inferrer_constructor": infer_scalar_ge,
    "grad_transform": bprop_scalar_ge,
}
