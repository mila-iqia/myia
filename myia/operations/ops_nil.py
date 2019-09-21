"""Operations on None."""

from .. import operations
from ..lib import core
from .utils import to_opdef


@to_opdef
@core
def nil_eq(a, b):
    """Implementation of `equal` (only use with Nil types)."""
    return a is None and b is None


@to_opdef
@core
def nil_ne(a, b):
    """Implementation of `not_equal` (only use with Nil types)."""
    return not operations.nil_eq(a, b)


@to_opdef
@core
def nil_bool(x):
    """Converting Nil (None) to Bool returns False."""
    return False
