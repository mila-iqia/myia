"""Definitions for the primitive `partial`."""

from ..lib import (
    AbstractFunction,
    PartialApplication,
    Possibilities,
    standard_prim,
)
from . import primitives as P


def pyimpl_partial(f, *args):
    """Implement `partial`."""
    def res(*others):
        return f(*(args + others))
    return res


@standard_prim(P.partial)
async def infer_partial(self, engine, fn, *args):
    """Infer the return type of primitive `partial`."""
    fns = await fn.get()
    assert isinstance(fns, Possibilities)
    return AbstractFunction(*[
        PartialApplication(fn, list(args)) for fn in fns
    ])


__operation_defaults__ = {
    'name': 'partial',
    'registered_name': 'partial',
    'mapping': P.partial,
    'python_implementation': pyimpl_partial,
}


__primitive_defaults__ = {
    'name': 'partial',
    'registered_name': 'partial',
    'type': 'backend',
    'python_implementation': pyimpl_partial,
    'inferrer_constructor': infer_partial,
    'grad_transform': None,
}
