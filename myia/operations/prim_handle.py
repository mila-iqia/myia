"""Definitions for the primitive `handle`."""

from ..lib import AbstractHandle, standard_prim
from . import primitives as P


@standard_prim(P.handle)
async def infer_handle(self, engine, init):
    """Infer the return type of primitive `handle`."""
    return AbstractHandle(init)


__operation_defaults__ = {
    'name': 'handle',
    'registered_name': 'handle',
    'mapping': P.handle,
}


__primitive_defaults__ = {
    'name': 'handle',
    'registered_name': 'handle',
    'type': 'backend',
    'inferrer_constructor': infer_handle,
    'grad_transform': None,
}
