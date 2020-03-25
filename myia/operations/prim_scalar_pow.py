"""Definitions for the primitive `scalar_pow`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_pow(x: Number, y: Number) -> Number:
    """Implement `scalar_pow`."""
    assert_scalar(x, y)
    return x ** y


infer_scalar_pow = UniformPrimitiveInferrer.partial(
    prim=P.scalar_pow, impl=pyimpl_scalar_pow, infer_value=False
)


@bprop_to_grad_transform(P.scalar_pow)
def bprop_scalar_pow(x, y, out, dout):
    """Backpropagator for `scalar_pow`."""
    ym1 = P.scalar_sub(y, 1)
    yym1 = P.scalar_mul(y, P.scalar_pow(x, ym1))
    dx = P.scalar_mul(dout, yym1)
    dy = P.scalar_mul(dout, P.scalar_mul(P.scalar_log(x), out))
    return dx, dy


__operation_defaults__ = {
    "name": "scalar_pow",
    "registered_name": "scalar_pow",
    "mapping": P.scalar_pow,
    "python_implementation": pyimpl_scalar_pow,
}


__primitive_defaults__ = {
    "name": "scalar_pow",
    "registered_name": "scalar_pow",
    "type": "backend",
    "python_implementation": pyimpl_scalar_pow,
    "inferrer_constructor": infer_scalar_pow,
    "grad_transform": bprop_scalar_pow,
}
