"""Definitions for the primitive `dict_getitem`."""

from .. import lib, xtype
from ..lib import standard_prim
from . import primitives as P


@standard_prim(P.dict_getitem)
async def infer_dict_getitem(
    self, engine, arg: lib.AbstractDict, idx: xtype.String
):
    """Infer the return type of primitive `dict_getitem`."""
    idx_v = self.require_constant(idx, argnum=2, range=set(arg.entries.keys()))
    return arg.entries[idx_v]


__operation_defaults__ = {
    "name": "dict_getitem",
    "registered_name": "dict_getitem",
    "mapping": P.dict_getitem,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "dict_getitem",
    "registered_name": "dict_getitem",
    "type": "inference",
    "python_implementation": None,
    "inferrer_constructor": infer_dict_getitem,
    "grad_transform": None,
}
