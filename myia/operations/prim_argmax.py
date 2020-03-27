"""Definitions for the primitive `argmax`."""

from .. import lib, xtype
from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractScalar,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import zeros_like
from . import primitives as P


@standard_prim(P.argmax)
async def infer_argmax(
    self, engine, input: lib.AbstractArray, dim: lib.u64tup_typecheck
):
    """Infer the return type of primitive `argmax`."""
    shp = ()
    shp_inp = input.xshape()
    dim = tuple(
        self.require_constant(e, argnum=f'"1:dim[{edx}]"')
        for edx, e in enumerate(dim.elements)
    )
    shp = list(shp_inp)
    for d in dim:
        shp[d] = 1
    shp = tuple(shp)
    return type(input)(
        AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Int[64]}),
        {SHAPE: shp, TYPE: input.xtype()},
    )


@bprop_to_grad_transform(P.argmax)
def bprop_argmax(x, axis, out, dout):
    """Backpropagator for primitive `scalar_max`."""
    return (zeros_like(x), zeros_like(axis))


__operation_defaults__ = {
    "name": "argmax",
    "registered_name": "argmax",
    "mapping": P.argmax,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "argmax",
    "registered_name": "argmax",
    "type": "backend",
    "python_implementation": None,
    "inferrer_constructor": infer_argmax,
    "grad_transform": bprop_argmax,
}
