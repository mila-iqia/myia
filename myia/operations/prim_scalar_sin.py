"""Definitions for the primitive `scalar_sin`."""

import math

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_sin(x: Number) -> Number:
    """Implement `scalar_sin`."""
    assert_scalar(x)
    return math.sin(x)


infer_scalar_sin = UniformPrimitiveInferrer.partial(
    prim=P.scalar_sin,
    impl=pyimpl_scalar_sin,
    infer_value=False
)


__operation_defaults__ = {
    'name': 'scalar_sin',
    'registered_name': 'scalar_sin',
    'mapping': P.scalar_sin,
    'python_implementation': pyimpl_scalar_sin,
}


__primitive_defaults__ = {
    'name': 'scalar_sin',
    'registered_name': 'scalar_sin',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_sin,
    'inferrer_constructor': infer_scalar_sin,
    'grad_transform': None,
}
