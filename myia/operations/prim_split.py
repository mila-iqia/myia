"""Definitions for the primitive `split`."""

import numpy as np

from ..lib import (
    SHAPE,
    TYPE,
    AbstractTuple,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import concat, zeros_like
from . import primitives as P


def pyimpl_split(x, sections, dim):
    """Implement `splitenate`."""
    return np.split(x, sections, axis=dim)


@standard_prim(P.split)
async def infer_split(self, engine, x, sections, dim):
    """Infer the return type of primitive `split`."""
    sections_v = [e.xvalue() for e in sections.elements]
    x_shp_v = x.xshape()
    dim_v = dim.xvalue()

    shp_r = ()
    for s in sections_v:
        shp_r = shp_r + (x_shp_v[:dim_v] + (s,) + x_shp_v[dim_v + 1:],)

    return AbstractTuple([
        type(x)(x.element, {SHAPE: out_shape, TYPE: x.xtype()})
        for out_shape in shp_r
    ])


@bprop_to_grad_transform(P.split)
def bprop_split(x, sections, dim, out, dout):
    """Backpropagator for primitive `split`."""
    x_grad = concat(dout, dim)
    return (x_grad, zeros_like(sections), zeros_like(dim))


__operation_defaults__ = {
    'name': 'split',
    'registered_name': 'split',
    'mapping': P.split,
    'python_implementation': pyimpl_split,
}


__primitive_defaults__ = {
    'name': 'split',
    'registered_name': 'split',
    'type': 'backend',
    'python_implementation': pyimpl_split,
    'inferrer_constructor': infer_split,
    'grad_transform': bprop_split,
}
