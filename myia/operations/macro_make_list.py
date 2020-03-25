"""Implementation of the make_list operation."""

from ..lib import Cons, Empty, listof, macro
from . import primitives as P


@macro
async def make_list(info, *elems):
    """Create a list using Cons and Empty."""
    g = info.graph
    lst = g.apply(Empty)
    abstracts = await info.abstracts()
    if not abstracts:
        return lst
    restype = info.engine.abstract_merge(*abstracts)
    for arg in reversed(info.nodes()):
        lst = g.apply(Cons, arg, lst)
    return g.apply(P.unsafe_static_cast, lst, listof(restype))


__operation_defaults__ = {
    "name": "make_list",
    "registered_name": "make_list",
    "mapping": make_list,
    "python_implementation": None,
}
