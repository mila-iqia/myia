"""Definitions for the primitive `scalar_tanh`."""

import math

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_tanh(x: Number) -> Number:
    """Implement `scalar_tanh`."""
    assert_scalar(x)
    return math.tanh(x)


infer_scalar_tanh = UniformPrimitiveInferrer.partial(
    prim=P.scalar_tanh, impl=pyimpl_scalar_tanh, infer_value=False
)


@bprop_to_grad_transform(P.scalar_tanh)
def bprop_scalar_tanh(x, out, dout):
    """Backpropagator for `scalar_tanh`."""
    return (P.scalar_sub(dout, P.scalar_mul(dout, P.scalar_mul(out, out))),)


__operation_defaults__ = {
    "name": "scalar_tanh",
    "registered_name": "scalar_tanh",
    "mapping": P.scalar_tanh,
    "python_implementation": pyimpl_scalar_tanh,
}


__primitive_defaults__ = {
    "name": "scalar_tanh",
    "registered_name": "scalar_tanh",
    "type": "backend",
    "python_implementation": pyimpl_scalar_tanh,
    "inferrer_constructor": infer_scalar_tanh,
    "grad_transform": bprop_scalar_tanh,
}
