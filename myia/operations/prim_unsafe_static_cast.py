"""Definitions for the primitive `unsafe_static_cast`."""

from .. import lib
from ..lib import standard_prim
from . import primitives as P


def pyimpl_unsafe_static_cast(x, t):
    """Implement `unsafe_static_cast`."""
    return x


@standard_prim(P.unsafe_static_cast)
async def infer_unsafe_static_cast(self, engine, x, typ: lib.AbstractType):
    """Infer the return type of primitive `unsafe_static_cast`."""
    return typ.xvalue()


__operation_defaults__ = {
    'name': 'unsafe_static_cast',
    'registered_name': 'unsafe_static_cast',
    'mapping': P.unsafe_static_cast,
    'python_implementation': pyimpl_unsafe_static_cast,
}


__primitive_defaults__ = {
    'name': 'unsafe_static_cast',
    'registered_name': 'unsafe_static_cast',
    'type': 'backend',
    'python_implementation': pyimpl_unsafe_static_cast,
    'inferrer_constructor': infer_unsafe_static_cast,
    'grad_transform': None,
}
