"""Definitions for the primitive `casttag`."""

from .. import lib, xtype
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import typeof, zeros_like
from . import primitives as P


def pyimpl_casttag(x, tag):
    """Implement `casttag`."""
    return x.cast(tag)


@standard_prim(P.casttag)
async def infer_casttag(
    self, engine, x: lib.AbstractTaggedUnion, tag: xtype.Int[64]
):
    """Infer the return type of primitive `casttag`."""
    opts = await lib.force_pending(x.options)
    tag_v = self.require_constant(tag, argnum=2, range={i for i, _ in opts})
    for i, typ in opts:
        if i == tag_v:
            return typ
    raise AssertionError("Unreachable")


@bprop_to_grad_transform(P.casttag, ignore_values=False)
def bprop_casttag(x, t, out, dout):
    """Backpropagator for primitive `casttag`."""
    return (P.unsafe_static_cast(P.tagged(dout, t), typeof(x)), zeros_like(t))


__operation_defaults__ = {
    "name": "casttag",
    "registered_name": "casttag",
    "mapping": P.casttag,
    "python_implementation": pyimpl_casttag,
}


__primitive_defaults__ = {
    "name": "casttag",
    "registered_name": "casttag",
    "type": "backend",
    "python_implementation": pyimpl_casttag,
    "inferrer_constructor": infer_casttag,
    "grad_transform": bprop_casttag,
}
