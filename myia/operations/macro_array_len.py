"""Implementation of the 'array_len' operation."""

from .. import lib
from ..lib import MyiaTypeError, macro
from . import primitives as P


@macro
async def array_len(info, arr: lib.AbstractArray):
    """Implement len(array)."""
    shp = (await arr.get()).xshape()
    if len(shp) < 1:
        raise MyiaTypeError("0d arrays have no len")
    shp_expr = info.graph.apply(P.shape, info.nodes()[0])
    return info.graph.apply(P.tuple_getitem, shp_expr, 0)


__operation_defaults__ = {
    "name": "array_len",
    "registered_name": "array_len",
    "mapping": array_len,
    "python_implementation": len,
}
