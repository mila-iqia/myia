"""Definitions for the primitive `reshape`."""

import operator
from functools import reduce

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
from ..operations import reshape, shape, zeros_like
from . import primitives as P


def _prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


def pyimpl_reshape(v, shape):
    """Implement `reshape`."""
    return np.reshape(v, shape)


@standard_prim(P.reshape)
async def infer_reshape(self, engine, a: AbstractArray, _shp: u64tup_typecheck):
    """Infer the return type of primitive `reshape`."""
    shp = build_value(_shp, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await force_pending(a.xshape())
    if (
        all(s is not ANYTHING for s in shp)
        and all(s is not ANYTHING for s in a_shp)
        and _prod(shp) != _prod(a_shp)
    ):
        raise MyiaShapeError(
            "Cannot change the total number of elements " "in reshape"
        )
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@bprop_to_grad_transform(P.reshape, ignore_values=False)
def bprop_reshape(xs, shp, out, dout):
    """Backpropagator for primitive `reshape`."""
    return (reshape(dout, shape(xs)), zeros_like(shp))


__operation_defaults__ = {
    "name": "reshape",
    "registered_name": "reshape",
    "mapping": P.reshape,
    "python_implementation": pyimpl_reshape,
}


__primitive_defaults__ = {
    "name": "reshape",
    "registered_name": "reshape",
    "type": "backend",
    "python_implementation": pyimpl_reshape,
    "inferrer_constructor": infer_reshape,
    "grad_transform": bprop_reshape,
}
