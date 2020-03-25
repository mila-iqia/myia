"""Definitions for the primitive `bool_and`."""

from ..lib import UniformPrimitiveInferrer
from ..xtype import Bool
from . import primitives as P


def pyimpl_bool_and(x: Bool, y: Bool) -> Bool:
    """Implement `bool_and`."""
    assert x is True or x is False
    assert y is True or y is False
    return x and y


infer_bool_and = UniformPrimitiveInferrer.partial(
    prim=P.bool_and, impl=pyimpl_bool_and, infer_value=True
)


__operation_defaults__ = {
    "name": "bool_and",
    "registered_name": "bool_and",
    "mapping": P.bool_and,
    "python_implementation": pyimpl_bool_and,
}


__primitive_defaults__ = {
    "name": "bool_and",
    "registered_name": "bool_and",
    "type": "backend",
    "python_implementation": pyimpl_bool_and,
    "inferrer_constructor": infer_bool_and,
    "grad_transform": False,
}
