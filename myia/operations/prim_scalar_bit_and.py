"""Definitions for the primitive `scalar_bit_and` x & y."""

from ..lib import UniformPrimitiveInferrer, assert_scalar
from ..xtype import Integral
from . import primitives as P


def pyimpl_scalar_bit_and(x: Integral, y: Integral) -> Integral:
    """Implement `scalar_bit_and`."""
    assert_scalar(x, y)
    return x & y


infer_scalar_bit_and = UniformPrimitiveInferrer.partial(
    prim=P.scalar_bit_and, impl=pyimpl_scalar_bit_and, infer_value=False
)

__operation_defaults__ = {
    "name": "scalar_bit_and",
    "registered_name": "scalar_bit_and",
    "mapping": P.scalar_bit_and,
    "python_implementation": pyimpl_scalar_bit_and,
}

__primitive_defaults__ = {
    "name": "scalar_bit_and",
    "registered_name": "scalar_bit_and",
    "type": "backend",
    "python_implementation": pyimpl_scalar_bit_and,
    "inferrer_constructor": infer_scalar_bit_and,
    "grad_transform": None,
}
