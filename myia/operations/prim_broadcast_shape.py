"""Definitions for the primitive `broadcast_shape`."""

from .. import xtype
from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    AbstractTuple,
    MyiaShapeError,
    bprop_to_grad_transform,
    standard_prim,
    u64tup_typecheck,
)
from ..operations import zeros_like
from . import primitives as P


def pyimpl_broadcast_shape(shpx, shpy):
    """Implement `broadcast_shape`."""
    orig_shpx = shpx
    orig_shpy = shpy
    dlen = len(shpx) - len(shpy)
    if dlen < 0:
        shpx = (1,) * -dlen + shpx
    elif dlen > 0:
        shpy = (1,) * dlen + shpy
    assert len(shpx) == len(shpy)
    shp = []
    for a, b in zip(shpx, shpy):
        if a == 1:
            shp.append(b)
        elif b == 1:
            shp.append(a)
        elif a == ANYTHING:
            shp.append(b)
        elif b == ANYTHING:
            shp.append(a)
        elif a == b:
            shp.append(a)
        else:
            raise ValueError(
                f'Cannot broadcast shapes {orig_shpx} and {orig_shpy}.'
            )
    return tuple(shp)


@standard_prim(P.broadcast_shape)
async def infer_broadcast_shape(self, engine,
                                xs: u64tup_typecheck,
                                ys: u64tup_typecheck):
    """Infer the return type of primitive `broadcast_shape`."""
    shp_x = tuple(x.xvalue() for x in xs.elements)
    shp_y = tuple(y.xvalue() for y in ys.elements)
    elems = []
    try:
        res = pyimpl_broadcast_shape(shp_x, shp_y)
    except ValueError as e:
        raise MyiaShapeError(e.args[0])
    for n in res:
        elems.append(AbstractScalar({
            VALUE: n,
            TYPE: xtype.UInt[64],
        }))
    return AbstractTuple(elems)


@bprop_to_grad_transform(P.broadcast_shape)
def bprop_broadcast_shape(shp1, shp2, out, dout):
    """Backpropagator for primitive `broadcast_shape`."""
    return (zeros_like(shp1), zeros_like(shp2))


__operation_defaults__ = {
    'name': 'broadcast_shape',
    'registered_name': 'broadcast_shape',
    'mapping': P.broadcast_shape,
    'python_implementation': pyimpl_broadcast_shape,
}


__primitive_defaults__ = {
    'name': 'broadcast_shape',
    'registered_name': 'broadcast_shape',
    'type': 'backend',
    'python_implementation': pyimpl_broadcast_shape,
    'inferrer_constructor': infer_broadcast_shape,
    'grad_transform': bprop_broadcast_shape,
}
