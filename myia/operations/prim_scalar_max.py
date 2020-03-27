"""Definitions for the primitive `scalar_max`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..operations import zeros_like
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_max(x: Number, y: Number) -> Number:
    """Implement `scalar_max`."""
    assert_scalar(x, y)
    return max(x, y)


infer_scalar_max = UniformPrimitiveInferrer.partial(
    prim=P.scalar_max, impl=pyimpl_scalar_max, infer_value=False
)


@bprop_to_grad_transform(P.scalar_max)
def bprop_scalar_max(x, y, out, dout):
    """Backpropagator for `scalar_max`."""
    dx, dy = P.switch(
        P.scalar_eq(x, y),
        (dout, dout),
        P.switch(
            P.scalar_gt(x, y), (dout, zeros_like(y)), (zeros_like(x), dout)
        ),
    )
    return (dx, dy)


__operation_defaults__ = {
    "name": "scalar_max",
    "registered_name": "scalar_max",
    "mapping": P.scalar_max,
    "python_implementation": pyimpl_scalar_max,
}


__primitive_defaults__ = {
    "name": "scalar_max",
    "registered_name": "scalar_max",
    "type": "backend",
    "python_implementation": pyimpl_scalar_max,
    "inferrer_constructor": infer_scalar_max,
    "grad_transform": bprop_scalar_max,
}
