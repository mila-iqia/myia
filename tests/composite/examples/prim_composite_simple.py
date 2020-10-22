"""Definitions for the primitive `composite_simple`."""
from myia.lib import AbstractScalar, scalar_add, scalar_div, scalar_sub


def pyimpl_composite_simple(x):
    """Implement `composite_simple`."""
    return scalar_div(scalar_add(x, 2), scalar_sub(3, x))


async def infer_composite_simple(self, engine, x: AbstractScalar):
    """Infer the return type of primitive `composite_simple`."""
    return x
