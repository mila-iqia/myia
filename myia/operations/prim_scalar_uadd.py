"""Definitions for the primitive `scalar_uadd`."""

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_uadd(x: Number) -> Number:
    """Implement `scalar_uadd`."""
    assert_scalar(x)
    return x


infer_scalar_uadd = UniformPrimitiveInferrer.partial(
    prim=P.scalar_uadd, impl=pyimpl_scalar_uadd, infer_value=True
)


@bprop_to_grad_transform(P.scalar_uadd)
def bprop_scalar_uadd(x, out, dout):
    """Backpropagator for `scalar_uadd`."""
    return (dout,)


__operation_defaults__ = {
    "name": "scalar_uadd",
    "registered_name": "scalar_uadd",
    "mapping": P.scalar_uadd,
    "python_implementation": pyimpl_scalar_uadd,
}


__primitive_defaults__ = {
    "name": "scalar_uadd",
    "registered_name": "scalar_uadd",
    "type": "backend",
    "python_implementation": pyimpl_scalar_uadd,
    "inferrer_constructor": infer_scalar_uadd,
    "grad_transform": bprop_scalar_uadd,
}
