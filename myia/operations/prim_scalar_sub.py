"""Definitions for the primitive `scalar_sub`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_sub(x: Number, y: Number) -> Number:
    """Implement `scalar_sub`."""
    assert_scalar(x, y)
    return x - y


infer_scalar_sub = UniformPrimitiveInferrer.partial(
    prim=P.scalar_sub,
    impl=pyimpl_scalar_sub,
    infer_value=False
)


@bprop_to_grad_transform(P.scalar_sub)
def bprop_scalar_sub(x, y, out, dout):
    """Backpropagator for `scalar_sub`."""
    return (dout, P.scalar_usub(dout))


__operation_defaults__ = {
    'name': 'scalar_sub',
    'registered_name': 'scalar_sub',
    'mapping': P.scalar_sub,
    'python_implementation': pyimpl_scalar_sub,
}


__primitive_defaults__ = {
    'name': 'scalar_sub',
    'registered_name': 'scalar_sub',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_sub,
    'inferrer_constructor': infer_scalar_sub,
    'grad_transform': bprop_scalar_sub,
}
