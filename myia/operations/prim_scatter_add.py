"""Definitions for the primitive `scatter_add`."""

from .. import lib, xtype
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import gather, zeros_like
from . import primitives as P


@standard_prim(P.scatter_add)
async def infer_scatter_add(self, engine,
                            input: lib.AbstractArray,
                            dim: xtype.UInt[64],
                            index: lib.AbstractArray,
                            src: lib.AbstractArray):
    """Infer the return type of primitive `scatter_add`."""
    return input


@bprop_to_grad_transform(P.scatter_add)
def bprop_scatter_add(x, dim, index, src, out, dout):
    """Backpropagator for primitive `scatter_add`."""
    src_grad = gather(dout, dim, index)
    return (dout, zeros_like(dim), zeros_like(index), src_grad)


__operation_defaults__ = {
    'name': 'scatter_add',
    'registered_name': 'scatter_add',
    'mapping': P.scatter_add,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'scatter_add',
    'registered_name': 'scatter_add',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_scatter_add,
    'grad_transform': bprop_scatter_add,
}
