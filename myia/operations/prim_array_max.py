"""Definitions for the primitive `array_max`."""

import operator
from functools import reduce

from .. import lib
from ..lib import (
    SHAPE,
    TYPE,
    bprop_to_grad_transform,
    myia_static,
    standard_prim,
)
from ..operations import (
    argmax,
    invert_permutation,
    reshape,
    scatter,
    shape,
    transpose,
    zeros_like,
)
from . import primitives as P


@standard_prim(P.array_max)
async def infer_array_max(
    self, engine, input: lib.AbstractArray, dim: lib.u64tup_typecheck
):
    """Infer the return type of primitive `array_max`."""
    shp = ()
    shp_inp = input.xshape()
    dim = tuple(
        self.require_constant(e, argnum=f'"1:dim[{edx}]"')
        for edx, e in enumerate(dim.elements)
    )
    shp = list(shp_inp)
    for d in dim:
        shp[d] = 1
    shp = tuple(shp)
    return type(input)(input.element, {SHAPE: shp, TYPE: input.xtype()})


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


@myia_static
def _dim_permute(d, xs):
    n = ()
    for _s in range(len(xs)):
        if _s not in d:
            n = n + (_s,)
    n = n + d
    return n


@myia_static
def _dim_reshape(d, xs):
    end = -len(d)
    ns = xs[:end] + (prod(xs[end:]),)
    return ns


@myia_static
def _last_dim(x):
    return len(x) - 1


@bprop_to_grad_transform(P.array_max)
def bprop_array_max(x, axis, out, dout):
    """Backpropagator for primitive `array_max`."""
    z = zeros_like(x)
    am = argmax(x, axis)

    n = _dim_permute(axis, shape(x))
    z = transpose(z, n)
    zs1 = shape(z)
    ns = _dim_reshape(axis, shape(z))
    z = reshape(z, ns)

    n_am = _dim_permute(axis, shape(am))
    am = transpose(am, n_am)
    ns_am = _dim_reshape(axis, shape(am))
    am = reshape(am, ns_am)

    n_dout = _dim_permute(axis, shape(dout))
    dout = transpose(dout, n_dout)
    ns_dout = _dim_reshape(axis, shape(dout))
    dout = reshape(dout, ns_dout)

    z = scatter(z, _last_dim(shape(z)), am, dout)
    z = reshape(z, zs1)
    z = transpose(z, invert_permutation(n))
    return (z, zeros_like(axis))


__operation_defaults__ = {
    "name": "array_max",
    "registered_name": "array_max",
    "mapping": P.array_max,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "array_max",
    "registered_name": "array_max",
    "type": "backend",
    "python_implementation": None,
    "inferrer_constructor": infer_array_max,
    "grad_transform": bprop_array_max,
}
