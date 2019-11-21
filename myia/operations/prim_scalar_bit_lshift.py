"""Definitions for the primitive `scalar_bit_lshift` x << y."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Integral
from . import primitives as P


def pyimpl_scalar_bit_lshift(x: Integral, y: Integral) -> Integral:
    """Implement `scalar_bit_lshift`."""
    assert_scalar(x, y)
    return x << y


infer_scalar_bit_lshift = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_lshift,
    impl=pyimpl_scalar_bit_lshift,
    infer_value=False
)

__operation_defaults__ = {
    'name': 'scalar_bit_lshift',
    'registered_name': 'scalar_bit_lshift',
    'mapping': P.scalar_bit_lshift,
    'python_implementation': pyimpl_scalar_bit_lshift,
}

__primitive_defaults__ = {
    'name': 'scalar_bit_lshift',
    'registered_name': 'scalar_bit_lshift',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_bit_lshift,
    'inferrer_constructor': infer_scalar_bit_lshift,
    'grad_transform': None,
}
