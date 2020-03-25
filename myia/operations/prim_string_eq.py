"""Definitions for the primitive `string_eq`."""

from ..lib import UniformPrimitiveInferrer
from ..xtype import Bool, String
from . import primitives as P


def pyimpl_string_eq(x: String, y: String) -> Bool:
    """Implement `string_eq`."""
    return x == y


infer_string_eq = UniformPrimitiveInferrer.partial(
    prim=P.string_eq, impl=pyimpl_string_eq, infer_value=True
)


__operation_defaults__ = {
    "name": "string_eq",
    "registered_name": "string_eq",
    "mapping": P.string_eq,
    "python_implementation": pyimpl_string_eq,
}


__primitive_defaults__ = {
    "name": "string_eq",
    "registered_name": "string_eq",
    "type": "inference",
    "python_implementation": pyimpl_string_eq,
    "inferrer_constructor": infer_string_eq,
    "grad_transform": None,
}
