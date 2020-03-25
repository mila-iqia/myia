"""Definitions for the primitive `scalar_usub`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_usub(x: Number) -> Number:
    """Implement `scalar_usub`."""
    assert_scalar(x)
    return -x


infer_scalar_usub = UniformPrimitiveInferrer.partial(
    prim=P.scalar_usub, impl=pyimpl_scalar_usub, infer_value=True
)


@bprop_to_grad_transform(P.scalar_usub)
def bprop_scalar_usub(x, out, dout):
    """Backpropagator for `scalar_usub`."""
    return (P.scalar_usub(dout),)


__operation_defaults__ = {
    "name": "scalar_usub",
    "registered_name": "scalar_usub",
    "mapping": P.scalar_usub,
    "python_implementation": pyimpl_scalar_usub,
}


__primitive_defaults__ = {
    "name": "scalar_usub",
    "registered_name": "scalar_usub",
    "type": "backend",
    "python_implementation": pyimpl_scalar_usub,
    "inferrer_constructor": infer_scalar_usub,
    "grad_transform": bprop_scalar_usub,
}
