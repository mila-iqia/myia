"""Definitions for the primitive `dict_setitem`."""

from .. import lib, xtype
from ..lib import standard_prim
from . import primitives as P


@standard_prim(P.dict_setitem)
async def infer_dict_setitem(self, engine,
                             arg: lib.AbstractDict,
                             idx: xtype.String,
                             value):
    """Infer the return type of primitive `dict_setitem`."""
    idx_v = self.require_constant(idx, argnum=2, range=set(arg.entries.keys()))
    return type(arg)({**arg.entries, idx_v: value})


__operation_defaults__ = {
    'name': 'dict_setitem',
    'registered_name': 'dict_setitem',
    'mapping': P.dict_setitem,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'dict_setitem',
    'registered_name': 'dict_setitem',
    'type': 'inference',
    'python_implementation': None,
    'inferrer_constructor': infer_dict_setitem,
    'grad_transform': None,
}
