"""Definitions for the primitive `scalar_mul`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_mul(x: Number, y: Number) -> Number:
    """Implement `scalar_mul`."""
    assert_scalar(x, y)
    return x * y


infer_scalar_mul = UniformPrimitiveInferrer.partial(
    prim=P.scalar_mul, impl=pyimpl_scalar_mul, infer_value=False
)


@bprop_to_grad_transform(P.scalar_mul)
def bprop_scalar_mul(x, y, out, dout):
    """Backpropagator for `scalar_mul`."""
    return (P.scalar_mul(dout, y), P.scalar_mul(dout, x))


__operation_defaults__ = {
    "name": "scalar_mul",
    "registered_name": "scalar_mul",
    "mapping": P.scalar_mul,
    "python_implementation": pyimpl_scalar_mul,
}


__primitive_defaults__ = {
    "name": "scalar_mul",
    "registered_name": "scalar_mul",
    "type": "backend",
    "python_implementation": pyimpl_scalar_mul,
    "inferrer_constructor": infer_scalar_mul,
    "grad_transform": bprop_scalar_mul,
}
