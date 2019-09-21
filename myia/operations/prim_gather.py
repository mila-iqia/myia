"""Definitions for the primitive `gather`."""

from .. import lib, xtype
from ..lib import SHAPE, TYPE, bprop_to_grad_transform, standard_prim
from ..operations import scatter_add, zeros_like
from . import primitives as P


@standard_prim(P.gather)
async def infer_gather(self, engine,
                       input: lib.AbstractArray,
                       dim: xtype.UInt[64],
                       index: lib.AbstractArray):
    """Infer the return type of primitive `gather`."""
    return type(input)(
        input.element,
        {SHAPE: index.xshape(), TYPE: input.xtype()}
    )


@bprop_to_grad_transform(P.gather)
def bprop_gather(x, dim, index, out, dout):
    """Backpropagator for primitive `gather`."""
    z = zeros_like(x)
    z = scatter_add(z, dim, index, dout)
    return (z, zeros_like(dim), zeros_like(index))


__operation_defaults__ = {
    'name': 'gather',
    'registered_name': 'gather',
    'mapping': P.gather,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'gather',
    'registered_name': 'gather',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_gather,
    'grad_transform': bprop_gather,
}
