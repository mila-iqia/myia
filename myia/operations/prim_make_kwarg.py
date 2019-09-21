"""Definitions for the primitive `make_kwarg`."""

from ..lib import AbstractKeywordArgument, standard_prim
from . import primitives as P


@standard_prim(P.make_kwarg)
async def infer_make_kwarg(self, engine, key, value):
    """Infer the return type of primitive `make_kwarg`."""
    k = key.xvalue()
    assert isinstance(k, str)
    return AbstractKeywordArgument(k, value)


__operation_defaults__ = {
    'name': 'make_kwarg',
    'registered_name': 'make_kwarg',
    'mapping': P.make_kwarg,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'make_kwarg',
    'registered_name': 'make_kwarg',
    'type': 'inference',
    'python_implementation': None,
    'inferrer_constructor': infer_make_kwarg,
    'grad_transform': None,
}
