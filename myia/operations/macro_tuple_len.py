"""Implementation of the 'tuple_len' operation."""

from .. import lib
from ..lib import Constant, macro


@macro
async def tuple_len(info, tup: lib.AbstractTuple):
    """Implement len(tuple)."""
    return Constant(len((await tup.get()).elements))


__operation_defaults__ = {
    "name": "tuple_len",
    "registered_name": "tuple_len",
    "mapping": tuple_len,
    "python_implementation": len,
}
