"""Definitions for the primitive `transpose`."""

import numpy as np

from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    MyiaShapeError,
    bprop_to_grad_transform,
    build_value,
    force_pending,
    standard_prim,
    u64tup_typecheck,
)
from ..operations import invert_permutation, transpose, zeros_like
from . import primitives as P


def pyimpl_transpose(v, permutation):
    """Implement `transpose`."""
    return np.transpose(v, permutation)


@standard_prim(P.transpose)
async def infer_transpose(self, engine,
                          a: AbstractArray, permutation: u64tup_typecheck):
    """Infer the return type of primitive `transpose`."""
    perm = build_value(permutation, default=ANYTHING)
    if perm == ANYTHING:
        shp = (ANYTHING,) * len(permutation.elements)
    else:
        a_shp = await force_pending(a.xshape())
        if list(sorted(perm)) != list(range(len(a_shp))):
            raise MyiaShapeError(
                'The second argument of transpose must be a permutation of'
                ' all of the array\'s axes.',
            )

        shp = tuple(a_shp[i] for i in perm)
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@bprop_to_grad_transform(P.transpose)
def bprop_transpose(xs, perm, out, dout):
    """Backpropagator for primitive `transpose`."""
    return (transpose(dout, invert_permutation(perm)),
            zeros_like(perm))


__operation_defaults__ = {
    'name': 'transpose',
    'registered_name': 'transpose',
    'mapping': P.transpose,
    'python_implementation': pyimpl_transpose,
}


__primitive_defaults__ = {
    'name': 'transpose',
    'registered_name': 'transpose',
    'type': 'backend',
    'python_implementation': pyimpl_transpose,
    'inferrer_constructor': infer_transpose,
    'grad_transform': bprop_transpose,
}
