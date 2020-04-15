"""Definitions for the primitive `make_handle`."""

from .. import xtype
from ..lib import AbstractHandle, AbstractType, standard_prim
from . import primitives as P


@standard_prim(P.make_handle)
async def infer_make_handle(self, engine, handle_id: xtype.Int[64], init):
    """Infer the return type of primitive `make_handle`."""
    return AbstractHandle(init)


__operation_defaults__ = {
    "name": "make_handle",
    "registered_name": "make_handle",
    "mapping": P.make_handle,
}


__primitive_defaults__ = {
    "name": "make_handle",
    "registered_name": "make_handle",
    "type": "backend",
    "inferrer_constructor": infer_make_handle,
    "grad_transform": None,
}
