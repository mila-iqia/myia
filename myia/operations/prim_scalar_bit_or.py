"""Definitions for the primitive `scalar_bit_or` x | y."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Integral
from . import primitives as P


def pyimpl_scalar_bit_or(x: Integral, y: Integral) -> Integral:
    """Implement `scalar_bit_or`."""
    assert_scalar(x, y)
    return x | y


infer_scalar_bit_or = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_or,
    impl=pyimpl_scalar_bit_or,
    infer_value=False
)

__operation_defaults__ = {
    'name': 'scalar_bit_or',
    'registered_name': 'scalar_bit_or',
    'mapping': P.scalar_bit_or,
    'python_implementation': pyimpl_scalar_bit_or,
}

__primitive_defaults__ = {
    'name': 'scalar_bit_or',
    'registered_name': 'scalar_bit_or',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_bit_or,
    'inferrer_constructor': infer_scalar_bit_or,
    'grad_transform': None,
}
