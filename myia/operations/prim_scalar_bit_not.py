"""Definitions for the primitive `scalar_bit_not` ~x."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Integral
from . import primitives as P


def pyimpl_scalar_bit_not(x: Integral) -> Integral:
    """Implement `scalar_bit_not`."""
    assert_scalar(x)
    return ~x


infer_scalar_bit_not = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_not,
    impl=pyimpl_scalar_bit_not,
    infer_value=False
)

__operation_defaults__ = {
    'name': 'scalar_bit_not',
    'registered_name': 'scalar_bit_not',
    'mapping': P.scalar_bit_not,
    'python_implementation': pyimpl_scalar_bit_not,
}

__primitive_defaults__ = {
    'name': 'scalar_bit_not',
    'registered_name': 'scalar_bit_not',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_bit_not,
    'inferrer_constructor': infer_scalar_bit_not,
    'grad_transform': None,
}
