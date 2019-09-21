"""Definitions for the primitive `scalar_tan`."""

import math

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_tan(x: Number) -> Number:
    """Implement `scalar_tan`."""
    assert_scalar(x)
    return math.tan(x)


infer_scalar_tan = UniformPrimitiveInferrer.partial(
    prim=P.scalar_tan,
    impl=pyimpl_scalar_tan,
    infer_value=False
)


__operation_defaults__ = {
    'name': 'scalar_tan',
    'registered_name': 'scalar_tan',
    'mapping': P.scalar_tan,
    'python_implementation': pyimpl_scalar_tan,
}


__primitive_defaults__ = {
    'name': 'scalar_tan',
    'registered_name': 'scalar_tan',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_tan,
    'inferrer_constructor': infer_scalar_tan,
    'grad_transform': None,
}
