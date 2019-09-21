"""Definitions for the primitive `identity`."""

from ..lib import bprop_to_grad_transform, standard_prim
from . import primitives as P


def pyimpl_identity(x):
    """Implement `identity`."""
    return x


@standard_prim(P.identity)
async def infer_identity(self, engine, x):
    """Infer the return type of primitive `identity`."""
    return x


@bprop_to_grad_transform(P.identity)
def bprop_identity(x, out, dout):
    """Backpropagator for primitive `identity`."""
    return (dout,)


__operation_defaults__ = {
    'name': 'identity',
    'registered_name': 'identity',
    'mapping': P.identity,
    'python_implementation': pyimpl_identity,
}


__primitive_defaults__ = {
    'name': 'identity',
    'registered_name': 'identity',
    'type': 'backend',
    'python_implementation': pyimpl_identity,
    'inferrer_constructor': infer_identity,
    'grad_transform': None,
}
