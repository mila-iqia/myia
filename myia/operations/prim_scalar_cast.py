"""Definitions for the primitive `scalar_cast`."""

import numpy as np

from .. import lib, xtype
from ..lib import (
    TYPE,
    AbstractScalar,
    MyiaTypeError,
    bprop_to_grad_transform,
    force_pending,
    standard_prim,
    type_to_abstract,
)
from ..operations import typeof
from . import primitives as P


def pyimpl_scalar_cast(x, t):
    """Implement `scalar_cast`."""
    t = type_to_abstract(t)
    assert isinstance(t, AbstractScalar)
    t = t.values[TYPE]
    assert issubclass(t, xtype.Number)
    dtype = xtype.type_to_np_dtype(t)
    return getattr(np, dtype)(x)


@standard_prim(P.scalar_cast)
async def infer_scalar_cast(self, engine, scalar, typ: lib.AbstractType):
    """Infer the return type of primitive `scalar_cast`."""
    a = typ.element
    if not isinstance(a, AbstractScalar):
        raise MyiaTypeError('scalar_cast must cast to a scalar type')
    t = a.xtype()
    engine.check(xtype.Number, t)
    engine.check(xtype.Number, await force_pending(scalar.xtype()))
    values = {**scalar.values, TYPE: t}
    return lib.AbstractScalar(values)


@bprop_to_grad_transform(P.scalar_cast)
def bprop_scalar_cast(x, t, out, dout):
    """Backpropagator for primitive `scalar_cast`."""
    return (P.scalar_cast(dout, typeof(x)), t)


__operation_defaults__ = {
    'name': 'scalar_cast',
    'registered_name': 'scalar_cast',
    'mapping': P.scalar_cast,
    'python_implementation': pyimpl_scalar_cast,
}


__primitive_defaults__ = {
    'name': 'scalar_cast',
    'registered_name': 'scalar_cast',
    'type': 'backend',
    'python_implementation': pyimpl_scalar_cast,
    'inferrer_constructor': infer_scalar_cast,
    'grad_transform': bprop_scalar_cast,
}
