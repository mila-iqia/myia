"""Definitions for the primitive `embedding`."""

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


def pyimpl_embedding(indices, weights):
    """Implement `embedding`."""
    return np.take(weights, indices, axis=0)


@standard_prim(P.embedding)
async def infer_embedding(self, engine,
                          indices: AbstractArray,
                          weights: AbstractArray):
    """Infer the return type of primitive `embedding`."""
    indices_shape = tuple(await force_pending(indices.xshape()))
    weights_shape = tuple(await force_pending(weights.xshape()))
    assert len(weights_shape) == 2
    output_shape = indices_shape + (weights_shape[1],)
    return AbstractArray(
        weights.element,
        {SHAPE: output_shape, TYPE: await force_pending(weights.xtype())}
    )


@bprop_to_grad_transform(P.embedding)
def bprop_embedding(indices, weights, out, dout):
    """Backpropagator for primitive `embedding`."""
    return (zeros_like(indices),
            P.grad_embedding_weights(indices, weights, dout))


__operation_defaults__ = {
    'name': 'embedding',
    'registered_name': 'embedding',
    'mapping': P.embedding,
    'python_implementation': pyimpl_embedding,
}


__primitive_defaults__ = {
    'name': 'embedding',
    'registered_name': 'embedding',
    'type': 'backend',
    'python_implementation': pyimpl_embedding,
    'inferrer_constructor': infer_embedding,
    'grad_transform': bprop_embedding,
}
