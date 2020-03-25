"""Definitions for the primitive `tuple_getitem`."""

from .. import lib, xtype
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import tuple_setitem, zeros_like
from . import primitives as P


def pyimpl_tuple_getitem(data, item):
    """Implement `getitem`."""
    return data[item]


def debugvm_tuple_getitem(vm, data, item):
    """Implement `getitem`."""
    return vm.convert(data[item])


@standard_prim(P.tuple_getitem)
async def infer_tuple_getitem(
    self, engine, arg: lib.AbstractTuple, idx: xtype.Int[64]
):
    """Infer the return type of primitive `tuple_getitem`."""
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    return arg.elements[idx_v]


@bprop_to_grad_transform(P.tuple_getitem, ignore_values=False)
def bprop_tuple_getitem(data, idx, out, dout):
    """Backpropagator for primitive `tuple_getitem`."""
    return (tuple_setitem(zeros_like(data), idx, dout), zeros_like(idx))


__operation_defaults__ = {
    "name": "tuple_getitem",
    "registered_name": "tuple_getitem",
    "mapping": P.tuple_getitem,
    "python_implementation": pyimpl_tuple_getitem,
}


__primitive_defaults__ = {
    "name": "tuple_getitem",
    "registered_name": "tuple_getitem",
    "type": "backend",
    "python_implementation": pyimpl_tuple_getitem,
    "debugvm_implementation": debugvm_tuple_getitem,
    "inferrer_constructor": infer_tuple_getitem,
    "grad_transform": bprop_tuple_getitem,
}
