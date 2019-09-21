"""Definitions for the primitive `scalar_trunc`."""

import math

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_trunc(x: Number) -> Number:
    """Implement `scalar_trunc`."""
    assert_scalar(x)
    return math.trunc(x)


infer_scalar_trunc = UniformPrimitiveInferrer.partial(
    prim=P.scalar_trunc,
    impl=pyimpl_scalar_trunc,
    infer_value=False
)


__operation_defaults__ = {
    'name': 'scalar_trunc',
    'registered_name': 'scalar_trunc',
    'mapping': P.scalar_trunc,
    'python_implementation': pyimpl_scalar_trunc,
}


__primitive_defaults__ = {
    'name': 'scalar_trunc',
    'registered_name': 'scalar_trunc',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_trunc,
    'inferrer_constructor': infer_scalar_trunc,
    'grad_transform': None,
}
