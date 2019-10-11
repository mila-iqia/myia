"""Definitions for the primitive `scalar_abs`."""

from ..operations import scalar_sign
from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_abs(x: Number) -> Number:
    """Implement `scalar_abs`."""
    assert_scalar(x)
    return abs(x)


infer_scalar_abs = UniformPrimitiveInferrer.partial(
    prim=P.scalar_abs,
    impl=pyimpl_scalar_abs,
    infer_value=False
)


@bprop_to_grad_transform(P.scalar_abs)
def bprop_scalar_abs(x, out, dout):  # pragma: no cover
    """Backpropagator for `scalar_abs`."""
    return (scalar_sign(x),)


__operation_defaults__ = {
    'name': 'scalar_abs',
    'registered_name': 'scalar_abs',
    'mapping': P.scalar_abs,
    'python_implementation': pyimpl_scalar_abs,
}


__primitive_defaults__ = {
    'name': 'scalar_abs',
    'registered_name': 'scalar_abs',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_abs,
    'inferrer_constructor': infer_scalar_abs,
    'grad_transform': bprop_scalar_abs,
}
