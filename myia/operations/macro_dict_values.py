"""Implementation of the 'dict_values' operation."""

from .. import lib
from ..lib import macro
from . import primitives as P


@macro
async def dict_values(info, d: lib.AbstractDict):
    """Implement dict.values()."""
    typ = await d.get()
    getters = [info.graph.apply(P.dict_getitem, d.node, k)
               for k in typ.entries]
    return info.graph.apply(P.make_tuple, *getters)


__operation_defaults__ = {
    'name': 'dict_values',
    'registered_name': 'dict_values',
    'mapping': dict_values,
    'python_implementation': None,
}
