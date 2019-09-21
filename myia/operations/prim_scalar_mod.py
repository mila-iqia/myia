"""Definitions for the primitive `scalar_mod`."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_mod(x: Number, y: Number) -> Number:
    """Implement `scalar_mod`."""
    assert_scalar(x, y)
    return x % y


infer_scalar_mod = UniformPrimitiveInferrer.partial(
    prim=P.scalar_mod,
    impl=pyimpl_scalar_mod,
    infer_value=False
)


__operation_defaults__ = {
    'name': 'scalar_mod',
    'registered_name': 'scalar_mod',
    'mapping': P.scalar_mod,
    'python_implementation': pyimpl_scalar_mod,
}


__primitive_defaults__ = {
    'name': 'scalar_mod',
    'registered_name': 'scalar_mod',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_mod,
    'inferrer_constructor': infer_scalar_mod,
    'grad_transform': None,
}
