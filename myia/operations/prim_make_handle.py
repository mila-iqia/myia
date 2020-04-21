"""Definitions for the primitive `make_handle`."""

from .. import xtype
from ..lib import (
    AbstractHandle,
    AbstractType,
    standard_prim,
    VALUE,
    TYPE,
    ANYTHING,
    AbstractTuple,
    AbstractScalar,
)
from . import primitives as P


@standard_prim(P.make_handle)
async def infer_make_handle(
    self, engine, typ: AbstractType, universe: xtype.UniverseType
):
    """Infer the return type of primitive `make_handle`."""
    return AbstractTuple(
        (
            AbstractScalar({VALUE: ANYTHING, TYPE: xtype.UniverseType}),
            AbstractHandle(typ.element),
        )
    )


__operation_defaults__ = {
    "name": "make_handle",
    "registered_name": "make_handle",
    "mapping": P.make_handle,
}


__primitive_defaults__ = {
    "name": "make_handle",
    "registered_name": "make_handle",
    "type": "backend",
    "universal": True,
    "inferrer_constructor": infer_make_handle,
    "grad_transform": None,
}
