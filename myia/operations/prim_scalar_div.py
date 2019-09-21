"""Definitions for the primitive `scalar_div`."""

import numpy as np

from ..lib import (
    UniformPrimitiveInferrer,
    assert_scalar,
    bprop_to_grad_transform,
)
from ..xtype import Number
from . import primitives as P


def pyimpl_scalar_div(x: Number, y: Number) -> Number:
    """Implement `scalar_div`."""
    assert_scalar(x, y)
    if isinstance(x, (float, np.floating)):
        return x / y
    else:
        return int(x / y)


infer_scalar_div = UniformPrimitiveInferrer.partial(
    prim=P.scalar_div,
    impl=pyimpl_scalar_div,
    infer_value=False
)


@bprop_to_grad_transform(P.scalar_div)
def bprop_scalar_div(x, y, out, dout):
    """Backpropagator for `scalar_div`."""
    return (P.scalar_div(dout, y),
            P.scalar_mul(P.scalar_usub(dout), P.scalar_div(out, y)))


__operation_defaults__ = {
    'name': 'scalar_div',
    'registered_name': 'scalar_div',
    'mapping': P.scalar_div,
    'python_implementation': pyimpl_scalar_div,
}


__primitive_defaults__ = {
    'name': 'scalar_div',
    'registered_name': 'scalar_div',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_div,
    'inferrer_constructor': infer_scalar_div,
    'grad_transform': bprop_scalar_div,
}
