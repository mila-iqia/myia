"""Definitions for the primitive `raise_`."""

from .. import xtype
from ..lib import AbstractBottom, bprop_to_grad_transform, standard_prim
from . import primitives as P


def pyimpl_raise(x):
    """Implement `raise_`."""
    raise x


@standard_prim(P.raise_)
async def infer_raise(self, engine, x: xtype.ExceptionType):
    """Infer the return type of primitive `raise_`."""
    return AbstractBottom()


@bprop_to_grad_transform(P.raise_)
def bprop_raise(x, out, dout):
    """Backpropagator for primitive `raise_`."""
    raise x


__operation_defaults__ = {
    "name": "raise",
    "registered_name": "raise_",
    "mapping": P.raise_,
    "python_implementation": pyimpl_raise,
}


__primitive_defaults__ = {
    "name": "raise",
    "registered_name": "raise_",
    "type": "backend",
    "python_implementation": pyimpl_raise,
    "inferrer_constructor": infer_raise,
    "grad_transform": bprop_raise,
}
