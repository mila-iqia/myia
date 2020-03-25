"""Definitions for the primitive `bool_eq`."""

from ..lib import UniformPrimitiveInferrer
from ..xtype import Bool
from . import primitives as P


def pyimpl_bool_eq(x: Bool, y: Bool) -> Bool:
    """Implement `bool_eq`."""
    assert x is True or x is False
    assert y is True or y is False
    return x == y


infer_bool_eq = UniformPrimitiveInferrer.partial(
    prim=P.bool_eq, impl=pyimpl_bool_eq, infer_value=True
)


__operation_defaults__ = {
    "name": "bool_eq",
    "registered_name": "bool_eq",
    "mapping": P.bool_eq,
    "python_implementation": pyimpl_bool_eq,
}


__primitive_defaults__ = {
    "name": "bool_eq",
    "registered_name": "bool_eq",
    "type": "backend",
    "python_implementation": pyimpl_bool_eq,
    "inferrer_constructor": infer_bool_eq,
    "grad_transform": False,
}
