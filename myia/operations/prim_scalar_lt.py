"""Definitions for the primitive `scalar_lt`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Bool, Number
from . import primitives as P


def pyimpl_scalar_lt(x: Number, y: Number) -> Bool:
    """Implement `scalar_lt`."""
    assert_scalar(x, y)
    return x < y


infer_scalar_lt = UniformPrimitiveInferrer.partial(
    prim=P.scalar_lt,
    impl=pyimpl_scalar_lt,
    infer_value=True
)


@bprop_to_grad_transform(P.scalar_lt)
def bprop_scalar_lt(x, y, out, dout):
    """Backpropagator for `scalar_lt`."""
    return (zeros_like(x), zeros_like(y))


__operation_defaults__ = {
    'name': 'scalar_lt',
    'registered_name': 'scalar_lt',
    'mapping': P.scalar_lt,
    'python_implementation': pyimpl_scalar_lt,
}


__primitive_defaults__ = {
    'name': 'scalar_lt',
    'registered_name': 'scalar_lt',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_lt,
    'inferrer_constructor': infer_scalar_lt,
    'grad_transform': bprop_scalar_lt,
}
