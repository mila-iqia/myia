"""Definitions for the primitive `universe_setitem`."""

from .. import lib, xtype
from ..lib import ANYTHING, TYPE, VALUE, AbstractScalar, standard_prim
from . import primitives as P


def pyimpl_universe_setitem(universe, handle, value):
    """Implement `universe_setitem`."""
    return universe.set(handle, value)


@standard_prim(P.universe_setitem)
async def infer_universe_setitem(self, engine,
                                 universe: xtype.UniverseType,
                                 handle: lib.AbstractHandle,
                                 value):
    """Infer the return type of primitive `universe_setitem`."""
    engine.abstract_merge(handle.element, value)
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: xtype.UniverseType,
    })


__operation_defaults__ = {
    'name': 'universe_setitem',
    'registered_name': 'universe_setitem',
    'mapping': P.universe_setitem,
    'python_implementation': pyimpl_universe_setitem,
}


__primitive_defaults__ = {
    'name': 'universe_setitem',
    'registered_name': 'universe_setitem',
    'type': 'backend',
    'python_implementation': pyimpl_universe_setitem,
    'inferrer_constructor': infer_universe_setitem,
    'grad_transform': None,
}
