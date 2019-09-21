"""Definitions for the primitive `return_`."""

from ..lib import standard_prim
from . import primitives as P


def pyimpl_return(x):
    """Implement `return_`."""
    return x


@standard_prim(P.return_)
async def infer_return_(self, engine, x):
    """Infer the return type of primitive `return_`."""
    return x


__operation_defaults__ = {
    'name': 'return',
    'registered_name': 'return_',
    'mapping': P.return_,
    'python_implementation': pyimpl_return,
}


__primitive_defaults__ = {
    'name': 'return',
    'registered_name': 'return_',
    'type': 'backend',
    'python_implementation': pyimpl_return,
    'inferrer_constructor': infer_return_,
    'grad_transform': None,
}
