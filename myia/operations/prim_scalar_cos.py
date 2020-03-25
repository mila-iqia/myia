"""Definitions for the primitive `scalar_cos`."""

import math

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_cos(x: Number) -> Number:
    """Implement `scalar_cos`."""
    assert_scalar(x)
    return math.cos(x)


infer_scalar_cos = UniformPrimitiveInferrer.partial(
    prim=P.scalar_cos, impl=pyimpl_scalar_cos, infer_value=False
)


__operation_defaults__ = {
    "name": "scalar_cos",
    "registered_name": "scalar_cos",
    "mapping": P.scalar_cos,
    "python_implementation": pyimpl_scalar_cos,
}


__primitive_defaults__ = {
    "name": "scalar_cos",
    "registered_name": "scalar_cos",
    "type": "backend",
    "python_implementation": pyimpl_scalar_cos,
    "inferrer_constructor": infer_scalar_cos,
    "grad_transform": None,
}
