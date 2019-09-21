"""Definitions for the primitive `scalar_floor`."""

import math

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_floor(x: Number) -> Number:
    """Implement `scalar_floor`."""
    assert_scalar(x)
    return math.floor(x)


infer_scalar_floor = UniformPrimitiveInferrer.partial(
    prim=P.scalar_floor,
    impl=pyimpl_scalar_floor,
    infer_value=False
)


__operation_defaults__ = {
    'name': 'scalar_floor',
    'registered_name': 'scalar_floor',
    'mapping': P.scalar_floor,
    'python_implementation': pyimpl_scalar_floor,
}


__primitive_defaults__ = {
    'name': 'scalar_floor',
    'registered_name': 'scalar_floor',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_floor,
    'inferrer_constructor': infer_scalar_floor,
    'grad_transform': None,
}
