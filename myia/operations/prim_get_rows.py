"""
Definition for primitive get_rows.

Inputs:
- the maximum number of indices n.
- a tensor of indices I with shape S.
- a tensor of values V with shape (*S, r)
Output:
- a matrix with shape (n, r), where each row i contains
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


def pyimpl_get_rows(nb_indices, indices, values):
    """Implement `get_rows`."""
    row_size = values.shape[-1]
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = np.zeros((nb_indices, row_size), dtype=values.dtype)
    for i in range(nb_indices):
        output[i] = (((broadcastable_indices == i) * values)
                     .reshape((-1, row_size))
                     .sum(axis=0))
    return output


@standard_prim(P.get_rows)
async def infer_get_rows(self, engine,
                         nb_indices: AbstractScalar,
                         indices: AbstractArray,
                         values: AbstractArray):
    """Infer the return type of primitive `get_rows`."""
    output_shape = (nb_indices.xvalue(),
                    (await force_pending(values.xshape()))[-1])
    return AbstractArray(values.element, {
        SHAPE: output_shape,
        TYPE: xtype.NDArray
    })


__operation_defaults__ = {
    'name': 'get_rows',
    'registered_name': 'get_rows',
    'mapping': P.get_rows,
    'python_implementation': pyimpl_get_rows,
}

__primitive_defaults__ = {
    'name': 'get_rows',
    'registered_name': 'get_rows',
    'type': 'backend',
    'python_implementation': pyimpl_get_rows,
    'inferrer_constructor': infer_get_rows,
    'grad_transform': None,
}
