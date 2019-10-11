"""Definitions for the primitive `concat`."""

import numpy as np

from ..lib import (
    SHAPE,
    TYPE,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import split, zeros_like
from . import primitives as P


def pyimpl_concat(x, dim):
    """Implement `concatenate`."""
    return np.concatenate(x, axis=dim)


@standard_prim(P.concat)
async def infer_concat(self, engine, x, dim):
    """Infer the return type of primitive `concat`."""
    """
    a_shp = a.xshape()
    b_shp = b.xshape()
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("concat needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in concat: {a_shp} and {b_shp}"
        )
    engine.abstract_merge(a.element, b.element)
    c_shp = (a_shp[0], b_shp[1])

    if a.xtype() != b.xtype():
        raise MyiaTypeError(
            f'Expect array of type {a.xtype()} '
            f'to have same type as array of type {b.xtype()}')
            #"""

    dim_v = dim.xvalue()

    new_dim_len = sum([e.xshape()[dim_v] for e in x.elements])

    shp_0 = x.elements[0].xshape()

    shp_f = shp_0[:dim_v] + (new_dim_len,) + shp_0[dim_v + 1:]

    return type(x.elements[0])(x.elements[0].element,
                               {SHAPE: shp_f, TYPE: x.elements[0].xtype()})


'''
@myia_static
def concat_grads(dout):
    return (dout[:, 0:4, :], dout[:, 4:9, :], dout[:, 9:, :])


@myia_static
def dummy(x):
    return x


@macro
async def _concat_grads_macro(info, dout_ref):
    dout = build_value(await dout_ref.get())
    # shp_explicit = ()

    ses = ((0, 4), (4, 9), (9, 15))

    x_grad = ()
    """
    for se in ses:
        x_grad = x_grad + \
            (array_getitem(dout, (0, se[0], 0), (3, se[1], 2), (1, 1, 1)),)
    """
    x_grad = x_grad + \
        (array_getitem(dout, (0, ses[0][0], 0), (3, ses[0][1], 2), (1, 1, 1)),)
    x_grad = x_grad + \
        (array_getitem(dout, (0, ses[1][0], 0), (3, ses[1][1], 2), (1, 1, 1)),)
    x_grad = x_grad + \
        (array_getitem(dout, (0, ses[2][0], 0), (3, ses[2][1], 2), (1, 1, 1)),)
    return Constant(x_grad)


@macro
async def _shps(info, x_ref):
    x = build_value(await x_ref.get())

    shps = ()
    for _x in x:
        shps = shps + (x.shape,)

    return Constant(shps)


@macro
async def _sections_m(info, shps_ref, dim_ref):
    shps = build_value(await shps_ref.get())
    dim = build_value(await dim_ref.get())

    sections = ()
    for _x in shps:
        sections = sections + (_x[dim],)

    return Constant(sections)


@myia_static
def _get_item(x, i):
    return x[i]
'''


@bprop_to_grad_transform(P.concat)
def bprop_concat(x, dim, out, dout):
    """Backpropagator for primitive `concat`."""

    """
    #return (zeros_like(x), zeros_like(dim))
    #idk = concat_grads(x, dim, dout)

    #return ((dout[:, 0:4, :], dout[:, 4:9, :], dout[:, 9:, :]),
             zeros_like(dim))

    #return (concat_grads(dout), zeros_like(dim))
    #return (x, zeros_like(dim))
    #return (x, dim)

    #ses = ((0,4),(4,9),(9,15))

    #x_grad = ()
    for se in ses:
        x_grad = x_grad +
            (array_getitem(dout, (0, se[0], 0), (3, se[1], 2), (1, 1, 1)),)

    x_grad0 = array_getitem(dout, (0, 0, 0), (3, 4, 2), (1, 1, 1))
    x_grad1 = array_getitem(dout, (0, 4, 0), (3, 9, 2), (1, 1, 1))
    x_grad2 = array_getitem(dout, (0, 9, 0), (3, 15, 2), (1, 1, 1))

    x_grad0 = array_getitem(
        dout, (0, ses[0][0], 0), (3, ses[0][1], 2), (1, 1, 1))
    x_grad1 = array_getitem(
        dout, (0, ses[1][0], 0), (3, ses[1][1], 2), (1, 1, 1))
    x_grad2 = array_getitem(
        dout, (0, ses[2][0], 0), (3, ses[2][1], 2), (1, 1, 1))

    #x_grad = (x_grad0, x_grad1, x_grad2)

    x_grad = x_grad + (array_getitem(dout, (0, 0, 0), (3, 4, 2), (1, 1, 1)),)
    x_grad = x_grad + (array_getitem(dout, (0, 4, 0), (3, 9, 2), (1, 1, 1)),)
    x_grad = x_grad + (array_getitem(dout, (0, 9, 0), (3, 15, 2), (1, 1, 1)),)

    x_grad = x_grad +
        (array_getitem(dout, (0, ses[0][0], 0), (3, ses[0][1], 2), (1, 1, 1)),)
    x_grad = x_grad +
        (array_getitem(dout, (0, ses[1][0], 0), (3, ses[1][1], 2), (1, 1, 1)),)
    x_grad = x_grad +
        (array_getitem(dout, (0, ses[2][0], 0), (3, ses[2][1], 2), (1, 1, 1)),)

    #x_grad = operations.array_cast(dout, operations.dtype(x))

    #return (not_hardcoded_version, zeros_like(dim))
    #x_grad = _concat_grads_macro(dout)


    #for se in range(3):
    #x_grad = x_grad + (dummy(x[0]),)
    #x_grad = x_grad + (dummy(x[1]),)
    #x_grad = x_grad + (dummy(x[2]),)

    #x_grad = list(x_grad)
    #x_grad = tuple(x_grad)

    def f(_x):
        x_grad = ()
        for i in range(3):
            x_grad = x_grad + (_x[i],)
        return x_grad


    def f(_x, _dim):
        sections = ()
        for __x in _x:
            sections = sections + (__x.shape[_dim],)
        return sections

    def f(_x):
        d = (4,5,6)
        sections = ()

        i = 0
        for __x in _x:
            #sections = sections + (d[i],)
            sections = sections + (_get_item(d, i),)
            i = i + 1
        return sections

    def f(_x, _dim):
        d = (4,5,6)
        sections = ()

        i = 0
        for __x in _x:
            #sections = sections + (d[i],)
            #sections = sections + (_get_item(d, i),)
            sections = sections + (tuple_getitem(d, i),)
            i = i + 1

        return sections
        '''
        #return (_x[0].shape[_dim], _x[1].shape[_dim], _x[2].shape[_dim])
        #tuple_setitem(
        x0 = tuple_getitem(_x, 0)
        x1 = tuple_getitem(_x, 1)
        x2 = tuple_getitem(_x, 2)
        return (x0.shape[_dim], x1.shape[_dim], x2.shape[_dim])
        #'''

    def f(_x, _dim):
        d = (4,5,6)
        sections = ()

        i = 0
        for __x in _x:
            #sections = sections + (d[i],)
            #sections = sections + (_get_item(d, i),)
            shp = _x.shape
            i = i + 1

        return sections

    def f(_x, _dim):
        sections = ()
        for __x in _x:
            #sections = sections + (__x.shape[_dim],)
            sections = sections + (_get_item(__x.shape, _dim),)
        return sections


    def f1(_x, _dim):
        #sections = ()
        _shp_s = _shps(_x)
        '''
        for __x in _shp_s:
            #sections = sections + (__x.shape[_dim],)
            #sections = sections + (_get_item(__x, _dim),)
            sections = sections + (__x.shape,)
        return sections
        '''
        return _shp_s


    def f1(_x, _dim):
        sections = ()
        for __x in _x:
            #sections = sections + (__x.shape[_dim],)
            sections = sections + (__x.shape,)
        return sections

    def f2(___shps, _dim):
        '''
        sections = ()
        for __x in shps:
            #sections = sections + (__x.shape[_dim],)
            #sections = sections + (__x[_dim],)
            #sections = sections + (_get_item(__x, _dim),)
            sections = sections + (__x[_dim],)
            '''
        sections = _sections_m(___shps, _dim)
        return sections

    #_shp__s = f1(x, dim)
    _shp__s = f1(x, dim)
    _sections = f2(_shp__s, dim)

    x_grad = split(dout, _sections, dim)
    return (x_grad, zeros_like(dim))
    #"""

    return (x, zeros_like(dim))


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
