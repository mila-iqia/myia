"""Definitions for the primitive `J`."""

from ..lib import (
    AbstractFunction,
    AbstractJTagged,
    JTransformedFunction,
    VirtualFunction2,
    bprop_to_grad_transform,
    standard_prim,
)
from ..monomorphize import type_fixer
from ..operations import Jinv
from . import primitives as P


@standard_prim(P.J)
async def infer_J(self, engine, x):
    """Infer the return type of primitive `J`."""
    if isinstance(x, AbstractFunction):
        v = await x.get()
        return AbstractFunction(*[JTransformedFunction(poss) for poss in v])
    elif isinstance(x, VirtualFunction2):
        vfn = type_fixer(None)(
            JTransformedFunction(VirtualFunction2(x.args, x.output))
        )
        assert isinstance(vfn, VirtualFunction2)
        return vfn
    else:
        return AbstractJTagged(x)


@bprop_to_grad_transform(P.J)
def bprop_J(x, out, dout):
    """Backpropagator for primitive `J`."""
    return (Jinv(dout),)


__operation_defaults__ = {
    "name": "J",
    "registered_name": "J",
    "mapping": P.J,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "J",
    "registered_name": "J",
    "type": "placeholder",
    "python_implementation": None,
    "inferrer_constructor": infer_J,
    "grad_transform": bprop_J,
}
