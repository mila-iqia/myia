"""Boolean operations."""

from .. import operations
from ..lib import core
from .utils import to_opdef


@to_opdef
@core(static_inline=True)
def not_(x):
    """Implementation of `not`."""
    return operations.bool_not(bool(x))


@to_opdef
@core(static_inline=True)
def bool_ne(x, y):
    """Implementation of `bool_ne`."""
    return operations.bool_not(operations.bool_eq(x, y))
