"""Definitions for the primitive `scalar_exp`."""

import math

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_exp(x: Number) -> Number:
    """Implement `scalar_exp`."""
    assert_scalar(x)
    return math.exp(x)


infer_scalar_exp = UniformPrimitiveInferrer.partial(
    prim=P.scalar_exp,
    impl=pyimpl_scalar_exp,
    infer_value=False
)


@bprop_to_grad_transform(P.scalar_exp)
def bprop_scalar_exp(x, out, dout):
    """Backpropagator for `scalar_exp`."""
    return (dout * out,)


__operation_defaults__ = {
    'name': 'scalar_exp',
    'registered_name': 'scalar_exp',
    'mapping': P.scalar_exp,
    'python_implementation': pyimpl_scalar_exp,
}


__primitive_defaults__ = {
    'name': 'scalar_exp',
    'registered_name': 'scalar_exp',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_exp,
    'inferrer_constructor': infer_scalar_exp,
    'grad_transform': bprop_scalar_exp,
}
