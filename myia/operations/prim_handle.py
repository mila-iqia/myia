"""Definitions for the primitive `handle`."""

from ..lib import AbstractHandle, HandleInstance, standard_prim
from . import primitives as P


def pyimpl_handle(init):
    """Implement `handle`."""
    return HandleInstance(init)


@standard_prim(P.handle)
async def infer_handle(self, engine, init):
    """Infer the return type of primitive `handle`."""
    return AbstractHandle(init)


__operation_defaults__ = {
    'name': 'handle',
    'registered_name': 'handle',
    'mapping': P.handle,
    'python_implementation': pyimpl_handle,
}


__primitive_defaults__ = {
    'name': 'handle',
    'registered_name': 'handle',
    'type': 'backend',
    'python_implementation': pyimpl_handle,
    'inferrer_constructor': infer_handle,
    'grad_transform': None,
}
