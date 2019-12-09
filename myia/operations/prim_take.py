"""Definitions for the primitive `take`."""

import numpy as np

from ..lib import (
    SHAPE,
    TYPE,
    AbstractArray,
    bprop_to_grad_transform,
    force_pending,
    standard_prim,
)
from ..operations import zeros_like
from . import primitives as P


def pyimpl_take(weights, indices):
    """Implement `take`."""
    return np.take(weights, indices, axis=0)


@standard_prim(P.take)
async def infer_take(self, engine,
                     weights: AbstractArray,
                     indices: AbstractArray):
    """Infer the return type of primitive `take`."""
    indices_shape = tuple(await force_pending(indices.xshape()))
    weights_shape = tuple(await force_pending(weights.xshape()))
    assert len(weights_shape) == 2
    output_shape = indices_shape + (weights_shape[1],)
    return AbstractArray(
        weights.element,
        {SHAPE: output_shape, TYPE: await force_pending(weights.xtype())}
    )


@bprop_to_grad_transform(P.take)
def bprop_take(weights, indices, out, dout):
    """Backpropagator for primitive `take`."""
    return (P.take_grad_weights(weights, indices, dout),
            zeros_like(indices))


__operation_defaults__ = {
    'name': 'take',
    'registered_name': 'take',
    'mapping': P.take,
    'python_implementation': pyimpl_take,
}


__primitive_defaults__ = {
    'name': 'take',
    'registered_name': 'take',
    'type': 'backend',
    'python_implementation': pyimpl_take,
    'inferrer_constructor': infer_take,
    'grad_transform': bprop_take,
}
