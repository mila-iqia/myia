"""Definitions for the primitive `scatter`."""

from .. import lib, xtype
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import gather, scatter, zeros_like
from . import primitives as P


@standard_prim(P.scatter)
async def infer_scatter(
    self,
    engine,
    input: lib.AbstractArray,
    dim: xtype.UInt[64],
    index: lib.AbstractArray,
    src: lib.AbstractArray,
):
    """Infer the return type of primitive `scatter`."""
    return input


@bprop_to_grad_transform(P.scatter)
def bprop_scatter(x, dim, index, src, out, dout):
    """Backpropagator for primitive `scatter`."""
    x_grad = scatter(dout, dim, index, zeros_like(src))
    src_grad = gather(dout, dim, index)
    return (x_grad, zeros_like(dim), zeros_like(index), src_grad)


__operation_defaults__ = {
    "name": "scatter",
    "registered_name": "scatter",
    "mapping": P.scatter,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "scatter",
    "registered_name": "scatter",
    "type": "backend",
    "python_implementation": None,
    "inferrer_constructor": infer_scatter,
    "grad_transform": bprop_scatter,
}
