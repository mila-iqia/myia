"""Definition for primitive take_grad_inp.

Internal primitive used to compute
gradient of primitive take wrt/ matrix input.

Inputs - the maximum number of indices n, a tensor of indices I with shape S,
and a tensor of values V with shape `(*S, r)`

Output - a matrix with shape (n, r), where each row i contains
the sum of rows in last dimension of V corresponding to
occurrences of i in last dimension of I. All other output
matrix rows that do not appear in indices are filled with
zero (0).
"""

import numpy as np

from .. import xtype
from ..lib import (
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractScalar,
    force_pending,
    standard_prim,
)
from . import primitives as P


def pyimpl_take_grad_inp(nb_indices, indices, values):
    """Implement `take_grad_inp`."""
    row_size = values.shape[-1]
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = np.zeros((nb_indices, row_size), dtype=values.dtype)
    for i in range(nb_indices):
        output[i] = (
            ((broadcastable_indices == i) * values)
            .reshape((-1, row_size))
            .sum(axis=0)
        )
    return output


@standard_prim(P.take_grad_inp)
async def infer_take_grad_inp(
    self,
    engine,
    nb_indices: AbstractScalar,
    indices: AbstractArray,
    values: AbstractArray,
):
    """Infer the return type of primitive `take_grad_inp`."""
    output_shape = (
        nb_indices.xvalue(),
        (await force_pending(values.xshape()))[-1],
    )
    return AbstractArray(
        values.element, {SHAPE: output_shape, TYPE: xtype.NDArray}
    )


__operation_defaults__ = {
    "name": "take_grad_inp",
    "registered_name": "take_grad_inp",
    "mapping": P.take_grad_inp,
    "python_implementation": pyimpl_take_grad_inp,
}

__primitive_defaults__ = {
    "name": "take_grad_inp",
    "registered_name": "take_grad_inp",
    "type": "backend",
    "python_implementation": pyimpl_take_grad_inp,
    "inferrer_constructor": infer_take_grad_inp,
    "grad_transform": None,
}
