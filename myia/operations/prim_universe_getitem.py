"""Definitions for the primitive `universe_getitem`."""

from .. import lib, xtype
from ..lib import broaden, standard_prim
from . import primitives as P


def pyimpl_universe_getitem(universe, handle):
    """Implement `universe_getitem`."""
    return universe.get(handle)


@standard_prim(P.universe_getitem)
async def infer_universe_getitem(
    self, engine, universe: xtype.UniverseType, handle: lib.AbstractHandle
):
    """Infer the return type of primitive `universe_getitem`."""
    return broaden(handle.element)


__operation_defaults__ = {
    "name": "universe_getitem",
    "registered_name": "universe_getitem",
    "mapping": P.universe_getitem,
    "python_implementation": pyimpl_universe_getitem,
}


__primitive_defaults__ = {
    "name": "universe_getitem",
    "registered_name": "universe_getitem",
    "type": "backend",
    "python_implementation": pyimpl_universe_getitem,
    "inferrer_constructor": infer_universe_getitem,
    "grad_transform": None,
}
