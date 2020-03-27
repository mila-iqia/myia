"""Definitions for the primitive `tuple_setitem`."""

from .. import lib, xtype
from ..lib import standard_prim
from . import primitives as P


def pyimpl_tuple_setitem(data, item, value):
    """Implement `tuple_setitem`."""
    return tuple(value if i == item else x for i, x in enumerate(data))


@standard_prim(P.tuple_setitem)
async def infer_tuple_setitem(
    self,
    engine,
    arg: lib.AbstractTuple,
    idx: xtype.Int[64],
    value: lib.AbstractValue,
):
    """Infer the return type of primitive `tuple_setitem`."""
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    elts = arg.elements
    new_elts = tuple([*elts[:idx_v], value, *elts[idx_v + 1 :]])
    return lib.AbstractTuple(new_elts)


__operation_defaults__ = {
    "name": "tuple_setitem",
    "registered_name": "tuple_setitem",
    "mapping": P.tuple_setitem,
    "python_implementation": pyimpl_tuple_setitem,
}


__primitive_defaults__ = {
    "name": "tuple_setitem",
    "registered_name": "tuple_setitem",
    "type": "backend",
    "python_implementation": pyimpl_tuple_setitem,
    "inferrer_constructor": infer_tuple_setitem,
    "grad_transform": None,
}
