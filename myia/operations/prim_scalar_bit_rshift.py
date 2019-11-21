"""Definitions for the primitive `scalar_bit_rshift` x >> y."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Integral
from . import primitives as P


def pyimpl_scalar_bit_rshift(x: Integral, y: Integral) -> Integral:
    """Implement `scalar_bit_rshift`."""
    assert_scalar(x, y)
    return x >> y


infer_scalar_bit_rshift = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_rshift,
    impl=pyimpl_scalar_bit_rshift,
    infer_value=False
)

__operation_defaults__ = {
    'name': 'scalar_bit_rshift',
    'registered_name': 'scalar_bit_rshift',
    'mapping': P.scalar_bit_rshift,
    'python_implementation': pyimpl_scalar_bit_rshift,
}

__primitive_defaults__ = {
    'name': 'scalar_bit_rshift',
    'registered_name': 'scalar_bit_rshift',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_bit_rshift,
    'inferrer_constructor': infer_scalar_bit_rshift,
    'grad_transform': None,
}
