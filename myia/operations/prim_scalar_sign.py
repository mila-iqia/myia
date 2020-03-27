"""Definitions for the primitive `scalar_sign`."""

import numpy as np

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_sign(x: Number) -> Number:
    """Implement `scalar_sign`."""
    assert_scalar(x)
    return np.sign(x)


infer_scalar_sign = UniformPrimitiveInferrer.partial(
    prim=P.scalar_sign, impl=pyimpl_scalar_sign, infer_value=False
)


@bprop_to_grad_transform(P.scalar_sign)
def bprop_scalar_sign(x, out, dout):  # pragma: no cover
    """Backpropagator for `scalar_sign`."""
    return (zeros_like(dout),)


__operation_defaults__ = {
    "name": "scalar_sign",
    "registered_name": "scalar_sign",
    "mapping": P.scalar_sign,
    "python_implementation": pyimpl_scalar_sign,
}


__primitive_defaults__ = {
    "name": "scalar_sign",
    "registered_name": "scalar_sign",
    "type": "backend",
    "python_implementation": pyimpl_scalar_sign,
    "inferrer_constructor": infer_scalar_sign,
    "grad_transform": bprop_scalar_sign,
}
