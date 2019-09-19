"""Definitions for the primitive `ioprint`."""

from .. import xtype
from ..lib import broaden, standard_prim
from . import primitives as P


def pyimpl_ioprint(iostate, obj):
    """Implement `ioprint`."""
    # The iostate exists in the type system to preserve order and purity, the
    # implementation doesn't have to do anything with it.
    print(obj)
    return iostate + 1


@standard_prim(P.ioprint)
async def infer_ioprint(self, engine, iostate: xtype.Int[64], obj):
    """Infer the return type of primitive `ioprint`."""
    return broaden(iostate)


__operation_defaults__ = {
    'name': 'ioprint',
    'registered_name': 'ioprint',
    'mapping': P.ioprint,
    'python_implementation': pyimpl_ioprint,
}


__primitive_defaults__ = {
    'name': 'ioprint',
    'registered_name': 'ioprint',
    'type': 'backend',
    'python_implementation': pyimpl_ioprint,
    'inferrer_constructor': infer_ioprint,
    'grad_transform': None,
}
