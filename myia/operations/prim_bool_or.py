"""Definitions for the primitive `bool_or`."""

from ..lib import UniformPrimitiveInferrer
from ..xtype import Bool
from . import primitives as P


def pyimpl_bool_or(x: Bool, y: Bool) -> Bool:
    """Implement `bool_or`."""
    assert x is True or x is False
    assert y is True or y is False
    return x or y


infer_bool_or = UniformPrimitiveInferrer.partial(
    prim=P.bool_or, impl=pyimpl_bool_or, infer_value=True
)


__operation_defaults__ = {
    "name": "bool_or",
    "registered_name": "bool_or",
    "mapping": P.bool_or,
    "python_implementation": pyimpl_bool_or,
}


__primitive_defaults__ = {
    "name": "bool_or",
    "registered_name": "bool_or",
    "type": "backend",
    "python_implementation": pyimpl_bool_or,
    "inferrer_constructor": infer_bool_or,
    "grad_transform": False,
}
