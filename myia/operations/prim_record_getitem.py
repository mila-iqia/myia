"""Definitions for the primitive `record_getitem`."""

from .. import lib, xtype
from ..lib import standard_prim
from . import primitives as P


@standard_prim(P.record_getitem)
async def infer_record_getitem(
    self, engine, data: lib.AbstractClassBase, attr: xtype.String
):
    """Infer the return type of primitive `record_getitem`."""
    attr_v = self.require_constant(attr, argnum=2)
    return data.attributes[attr_v]


__operation_defaults__ = {
    "name": "record_getitem",
    "registered_name": "record_getitem",
    "mapping": P.record_getitem,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "record_getitem",
    "registered_name": "record_getitem",
    "type": "inference",
    "python_implementation": None,
    "inferrer_constructor": infer_record_getitem,
    "grad_transform": None,
}
