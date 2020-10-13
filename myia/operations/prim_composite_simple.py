"""Definitions for the primitive `composite_simple`."""

from ..lib import (
    AbstractScalar,
    core,
    scalar_add,
    scalar_div,
    scalar_sub,
    standard_prim,
)
from . import primitives as P


@core
def pyimpl_composite_simple(x):
    """Implement `composite_simple`."""
    return scalar_div(scalar_add(x, 2), scalar_sub(3, x))


@standard_prim(P.composite_simple)
async def infer_composite_simple(self, engine, x: AbstractScalar):
    """Infer the return type of primitive `composite_simple`."""
    return x


__operation_defaults__ = {
    "name": "composite_simple",
    "registered_name": "composite_simple",
    "mapping": P.composite_simple,
    "python_implementation": pyimpl_composite_simple,
}


__primitive_defaults__ = {
    "name": "composite_simple",
    "registered_name": "composite_simple",
    "type": "composite",
    "python_implementation": pyimpl_composite_simple,
    "inferrer_constructor": infer_composite_simple,
    "grad_transform": None,
}
