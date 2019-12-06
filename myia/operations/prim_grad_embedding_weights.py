"""Definitions for the primitive `grad_embedding_weights`."""

import numpy as np

from ..lib import AbstractArray, standard_prim
from . import primitives as P


def pyimpl_grad_embedding_weights(indices, weights, dout):
    """Implement `grad_embedding_weights`."""
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = np.zeros(weights.shape, dtype=dout.dtype)
    for i in range(weights.shape[0]):
        output[i] = (((broadcastable_indices == i) * dout)
                     .reshape((-1, weights.shape[1]))
                     .sum(axis=0))
    return output


@standard_prim(P.grad_embedding_weights)
async def infer_grad_embedding_weights(self, engine,
                                       indices: AbstractArray,
                                       weights: AbstractArray,
                                       dout: AbstractArray):
    """Infer the return type of primitive `grad_embedding_weights`."""
    return weights


__operation_defaults__ = {
    'name': 'grad_embedding_weights',
    'registered_name': 'grad_embedding_weights',
    'mapping': P.grad_embedding_weights,
    'python_implementation': pyimpl_grad_embedding_weights,
}


__primitive_defaults__ = {
    'name': 'grad_embedding_weights',
    'registered_name': 'grad_embedding_weights',
    'type': 'backend',
    'python_implementation': pyimpl_grad_embedding_weights,
    'inferrer_constructor': infer_grad_embedding_weights,
    'grad_transform': None,
}
