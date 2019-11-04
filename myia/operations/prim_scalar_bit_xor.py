"""Definitions for the primitive `scalar_bit_xor` x ^ y."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_bit_xor(x: Number, y: Number) -> Number:
    """Implement `scalar_bit_xor`."""
    assert_scalar(x, y)
    return x ^ y


infer_scalar_bit_xor = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_xor,
    impl=pyimpl_scalar_bit_xor,
    infer_value=False
)

__operation_defaults__ = {
    'name': 'scalar_bit_xor',
    'registered_name': 'scalar_bit_xor',
    'mapping': P.scalar_bit_xor,
    'python_implementation': pyimpl_scalar_bit_xor,
}

__primitive_defaults__ = {
    'name': 'scalar_bit_xor',
    'registered_name': 'scalar_bit_xor',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_bit_xor,
    'inferrer_constructor': infer_scalar_bit_xor,
    'grad_transform': None,
}
