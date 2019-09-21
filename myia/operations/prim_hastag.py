"""Definitions for the primitive `hastag`."""

from .. import lib, xtype
from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import zeros_like
from . import primitives as P


def pyimpl_hastag(x, tag):
    """Implement `hastag`."""
    return x.has(tag)


@standard_prim(P.hastag)
async def infer_hastag(self, engine,
                       x: lib.AbstractTaggedUnion, tag: xtype.Int[64]):
    """Infer the return type of primitive `hastag`."""
    opts = await lib.force_pending(x.options)
    self.require_constant(
        tag, argnum=2,
        range={i for i, _ in opts}
    )
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: xtype.Bool,
    })


@bprop_to_grad_transform(P.hastag)
def bprop_hastag(x, t, out, dout):
    """Backpropagator for primitive `hastag`."""
    return (zeros_like(x), zeros_like(t))


__operation_defaults__ = {
    'name': 'hastag',
    'registered_name': 'hastag',
    'mapping': P.hastag,
    'python_implementation': pyimpl_hastag,
}


__primitive_defaults__ = {
    'name': 'hastag',
    'registered_name': 'hastag',
    'type': 'backend',
    'python_implementation': pyimpl_hastag,
    'inferrer_constructor': infer_hastag,
    'grad_transform': bprop_hastag,
}
