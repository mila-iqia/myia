"""Definitions for the primitive `concat`."""

import numpy as np

from ..lib import SHAPE, TYPE, bprop_to_grad_transform, standard_prim
from ..operations import split, zeros_like
from . import primitives as P


def pyimpl_concat(x, dim):
    """Implement `concatenate`."""
    return np.concatenate(x, axis=dim)


@standard_prim(P.concat)
async def infer_concat(self, engine, x, dim):
    """Infer the return type of primitive `concat`."""
    dim_v = dim.xvalue()

    new_dim_len = sum([e.xshape()[dim_v] for e in x.elements])

    shp_0 = x.elements[0].xshape()

    assert all(len(e.xshape()) == len(shp_0) for e in x.elements)

    for d in range(len(shp_0)):
        if d != dim_v:
            assert all(e.xshape()[d] == shp_0[d] for e in x.elements)

    shp_f = shp_0[:dim_v] + (new_dim_len,) + shp_0[dim_v + 1:]

    return type(x.elements[0])(x.elements[0].element,
                               {SHAPE: shp_f, TYPE: x.elements[0].xtype()})


@bprop_to_grad_transform(P.concat)
def bprop_concat(x, dim, out, dout):
    """Backpropagator for primitive `concat`."""
    def f(_x, _dim):
        sections = ()
        for __x in _x:
            sections = sections + (__x.shape[_dim],)
        return sections

    _sections = f(x, dim)
    x_grad = split(dout, _sections, dim)
    return (x_grad, zeros_like(dim))


__operation_defaults__ = {
    'name': 'concat',
    'registered_name': 'concat',
    'mapping': P.concat,
    'python_implementation': pyimpl_concat,
}


__primitive_defaults__ = {
    'name': 'concat',
    'registered_name': 'concat',
    'type': 'backend',
    'python_implementation': pyimpl_concat,
    'inferrer_constructor': infer_concat,
    'grad_transform': bprop_concat,
}
