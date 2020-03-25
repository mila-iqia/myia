"""Definitions for the primitive `scalar_log`."""

import math

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Float
from . import primitives as P


def pyimpl_scalar_log(x: Float) -> Float:
    """Implement `scalar_log`."""
    assert_scalar(x)
    return math.log(x)


infer_scalar_log = UniformPrimitiveInferrer.partial(
    prim=P.scalar_log, impl=pyimpl_scalar_log, infer_value=False
)


@bprop_to_grad_transform(P.scalar_log)
def bprop_scalar_log(x, out, dout):
    """Backpropagator for `scalar_log`."""
    return (dout / x,)


__operation_defaults__ = {
    "name": "scalar_log",
    "registered_name": "scalar_log",
    "mapping": P.scalar_log,
    "python_implementation": pyimpl_scalar_log,
}


__primitive_defaults__ = {
    "name": "scalar_log",
    "registered_name": "scalar_log",
    "type": "backend",
    "python_implementation": pyimpl_scalar_log,
    "inferrer_constructor": infer_scalar_log,
    "grad_transform": bprop_scalar_log,
}
