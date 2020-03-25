"""Definitions for the primitive `scalar_ne`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_ne(x: Number, y: Number) -> Bool:
    """Implement `scalar_ne`."""
    assert_scalar(x, y)
    return x != y


infer_scalar_ne = UniformPrimitiveInferrer.partial(
    prim=P.scalar_ne, impl=pyimpl_scalar_ne, infer_value=True
)


@bprop_to_grad_transform(P.scalar_ne)
def bprop_scalar_ne(x, y, out, dout):
    """Backpropagator for `scalar_ne`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    "name": "scalar_ne",
    "registered_name": "scalar_ne",
    "mapping": P.scalar_ne,
    "python_implementation": pyimpl_scalar_ne,
}


__primitive_defaults__ = {
    "name": "scalar_ne",
    "registered_name": "scalar_ne",
    "type": "backend",
    "python_implementation": pyimpl_scalar_ne,
    "inferrer_constructor": infer_scalar_ne,
    "grad_transform": bprop_scalar_ne,
}
