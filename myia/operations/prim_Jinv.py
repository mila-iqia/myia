"""Definitions for the primitive `Jinv`."""

from ..abstract.infer import compute_jinv_type
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import J
from . import primitives as P


@standard_prim(P.Jinv)
async def infer_Jinv(self, engine, x):
    """Infer the return type of primitive `Jinv`."""
    return await compute_jinv_type(x)


@bprop_to_grad_transform(P.Jinv)
def bprop_Jinv(x, out, dout):
    """Backpropagator for primitive `Jinv`."""
    return (J(dout),)


__operation_defaults__ = {
    "name": "Jinv",
    "registered_name": "Jinv",
    "mapping": P.Jinv,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "Jinv",
    "registered_name": "Jinv",
    "type": "placeholder",
    "python_implementation": None,
    "inferrer_constructor": infer_Jinv,
    "grad_transform": bprop_Jinv,
}
