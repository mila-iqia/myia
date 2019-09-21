"""Definitions for the primitive `scalar_add`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_add(x: Number, y: Number) -> Number:
    """Implement `scalar_add`."""
    assert_scalar(x, y)
    return x + y


infer_scalar_add = UniformPrimitiveInferrer.partial(
    prim=P.scalar_add,
    impl=pyimpl_scalar_add,
    infer_value=False
)


@bprop_to_grad_transform(P.scalar_add)
def bprop_scalar_add(x, y, out, dout):  # pragma: no cover
    """Backpropagator for `scalar_add`."""
    return (dout, dout)


__operation_defaults__ = {
    'name': 'scalar_add',
    'registered_name': 'scalar_add',
    'mapping': P.scalar_add,
    'python_implementation': pyimpl_scalar_add,
}


__primitive_defaults__ = {
    'name': 'scalar_add',
    'registered_name': 'scalar_add',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_add,
    'inferrer_constructor': infer_scalar_add,
    'grad_transform': bprop_scalar_add,
}
