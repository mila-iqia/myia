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


def pyimpl_take(inp, indices):
    """Implement `take`."""
    return np.take(inp, indices, axis=0)


@standard_prim(P.take)
async def infer_take(self, engine,
                     inp: AbstractArray,
                     indices: AbstractArray):
    """Infer the return type of primitive `take`."""
    indices_shape = tuple(await force_pending(indices.xshape()))
    inp_shape = tuple(await force_pending(inp.xshape()))
    assert len(inp_shape) == 2
    output_shape = indices_shape + (inp_shape[1],)
    return AbstractArray(
        inp.element,
        {SHAPE: output_shape, TYPE: await force_pending(inp.xtype())}
    )


@bprop_to_grad_transform(P.take)
def bprop_take(inp, indices, out, dout):
    """Backpropagator for primitive `take`."""
    return (P.take_grad_inp(P.shape(inp)[0], indices, dout),
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
