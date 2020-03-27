"""Definitions for the primitive `scalar_gt`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_gt(x: Number, y: Number) -> Bool:
    """Implement `scalar_gt`."""
    assert_scalar(x, y)
    return x > y


infer_scalar_gt = UniformPrimitiveInferrer.partial(
    prim=P.scalar_gt, impl=pyimpl_scalar_gt, infer_value=True
)


@bprop_to_grad_transform(P.scalar_gt)
def bprop_scalar_gt(x, y, out, dout):
    """Backpropagator for `scalar_gt`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    "name": "scalar_gt",
    "registered_name": "scalar_gt",
    "mapping": P.scalar_gt,
    "python_implementation": pyimpl_scalar_gt,
}


__primitive_defaults__ = {
    "name": "scalar_gt",
    "registered_name": "scalar_gt",
    "type": "backend",
    "python_implementation": pyimpl_scalar_gt,
    "inferrer_constructor": infer_scalar_gt,
    "grad_transform": bprop_scalar_gt,
}
