"""Definitions for the primitive `bool_not`."""

from ..lib import UniformPrimitiveInferrer
from ..xtype import Bool
from . import primitives as P


def pyimpl_bool_not(x: Bool) -> Bool:
    """Implement `bool_not`."""
    assert x is True or x is False
    return not x


infer_bool_not = UniformPrimitiveInferrer.partial(
    prim=P.bool_not,
    impl=pyimpl_bool_not,
    infer_value=True
)


__operation_defaults__ = {
    'name': 'bool_not',
    'registered_name': 'bool_not',
    'mapping': P.bool_not,
    'python_implementation': pyimpl_bool_not,
}


__primitive_defaults__ = {
    'name': 'bool_not',
    'registered_name': 'bool_not',
    'type': 'backend',
    'python_implementation': pyimpl_bool_not,
    'inferrer_constructor': infer_bool_not,
    'grad_transform': False,
}
