"""Implementation of the 'apply' operation."""

from .. import lib
from ..lib import MyiaTypeError, macro
from . import primitives as P


@macro
async def apply(info, fnref, *grouprefs):
    """Expand a varargs and keyword args call."""
    expanded = []
    g = info.graph
    for gref in grouprefs:
        t = await gref.get()
        if isinstance(t, lib.AbstractDict):
            for k in t.entries:
                extract = g.apply(P.dict_getitem, gref.node, k)
                mkkw = g.apply(P.make_kwarg, k, extract)
                expanded.append(mkkw)
        elif isinstance(t, lib.AbstractTuple):
            for i, _ in enumerate(t.elements):
                expanded.append(g.apply(P.tuple_getitem, gref.node, i))
        else:
            raise MyiaTypeError(
                'Can only expand tuple or dict in function application'
            )
    return g.apply(fnref.node, *expanded)


__operation_defaults__ = {
    'name': 'apply',
    'registered_name': 'apply',
    'mapping': apply,
    'python_implementation': None,
}
