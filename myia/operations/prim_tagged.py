"""Definitions for the primitive `tagged`."""

from .. import lib
from ..lib import TaggedValue, bprop_to_grad_transform, standard_prim
from ..operations import casttag, zeros_like
from . import primitives as P


def pyimpl_tagged(x, tag=None):
    """Implement `tagged`."""
    if tag is None:
        return x
    else:
        return TaggedValue(tag, x)


@standard_prim(P.tagged)
async def infer_tagged(self, engine, x, *rest):
    """Infer the return type of primitive `tagged`."""
    if len(rest) == 0:
        return lib.AbstractUnion([lib.broaden(x, loop=engine.loop)])
    elif len(rest) == 1:
        (tag,) = rest
        tag_v = self.require_constant(tag, argnum=2)
        return lib.AbstractTaggedUnion(
            [[tag_v, lib.broaden(x, loop=engine.loop)]]
        )
    else:
        raise lib.type_error_nargs(P.tagged, "1 or 2", len(rest) + 1)


@bprop_to_grad_transform(P.tagged)
def bprop_tagged(x, t, out, dout):
    """Backpropagator for primitive `tagged`."""
    return (casttag(dout, t), zeros_like(t))


__operation_defaults__ = {
    "name": "tagged",
    "registered_name": "tagged",
    "mapping": P.tagged,
    "python_implementation": pyimpl_tagged,
}


__primitive_defaults__ = {
    "name": "tagged",
    "registered_name": "tagged",
    "type": "backend",
    "python_implementation": pyimpl_tagged,
    "inferrer_constructor": infer_tagged,
    "grad_transform": None,
}
