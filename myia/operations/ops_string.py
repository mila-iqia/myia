"""String operations."""

from ..lib import core
from ..operations import bool_not, string_eq
from .utils import to_opdef


@to_opdef
@core
def string_ne(x, y):
    """Implementation of `string_ne`."""
    return bool_not(string_eq(x, y))
