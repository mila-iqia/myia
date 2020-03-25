"""Definitions for the primitive `scalar_eq`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_eq(x: Number, y: Number) -> Bool:
    """Implement `scalar_eq`."""
    assert_scalar(x, y)
    return x == y


infer_scalar_eq = UniformPrimitiveInferrer.partial(
    prim=P.scalar_eq, impl=pyimpl_scalar_eq, infer_value=True
)


@bprop_to_grad_transform(P.scalar_eq)
def bprop_scalar_eq(x, y, out, dout):
    """Backpropagator for `scalar_eq`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    "name": "scalar_eq",
    "registered_name": "scalar_eq",
    "mapping": P.scalar_eq,
    "python_implementation": pyimpl_scalar_eq,
}


__primitive_defaults__ = {
    "name": "scalar_eq",
    "registered_name": "scalar_eq",
    "type": "backend",
    "python_implementation": pyimpl_scalar_eq,
    "inferrer_constructor": infer_scalar_eq,
    "grad_transform": bprop_scalar_eq,
}
