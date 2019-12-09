"""Definitions for the primitive `take_grad_weights`."""

import numpy as np

from ..lib import AbstractArray, standard_prim
from . import primitives as P


def pyimpl_take_grad_weights(weights, indices, dout):
    """Implement `take_grad_weights`."""
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = np.zeros(weights.shape, dtype=dout.dtype)
    for i in range(weights.shape[0]):
        output[i] = (((broadcastable_indices == i) * dout)
                     .reshape((-1, weights.shape[1]))
                     .sum(axis=0))
    return output


@standard_prim(P.take_grad_weights)
async def infer_take_grad_weights(self, engine,
                                  weights: AbstractArray,
                                  indices: AbstractArray,
                                  dout: AbstractArray):
    """Infer the return type of primitive `take_grad_weights`."""
    return weights


__operation_defaults__ = {
    'name': 'take_grad_weights',
    'registered_name': 'take_grad_weights',
    'mapping': P.take_grad_weights,
    'python_implementation': pyimpl_take_grad_weights,
}


__primitive_defaults__ = {
    'name': 'take_grad_weights',
    'registered_name': 'take_grad_weights',
    'type': 'backend',
    'python_implementation': pyimpl_take_grad_weights,
    'inferrer_constructor': infer_take_grad_weights,
    'grad_transform': None,
}
