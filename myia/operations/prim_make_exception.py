"""Definitions for the primitive `exception`."""

from .. import xtype
from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    bprop_to_grad_transform,
    standard_prim,
)
from . import primitives as P


def pyimpl_make_exception(x):
    """Implement `make_exception`."""
    return Exception(x)


@standard_prim(P.make_exception)
async def infer_make_exception(self, engine, x):
    """Infer the return type of primitive `make_exception`."""
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.ExceptionType})


@bprop_to_grad_transform(P.make_exception)
def bprop_make_exception(x, out, dout):
    """Backpropagator for primitive `make_exception`."""
    return (x,)


__operation_defaults__ = {
    "name": "make_exception",
    "registered_name": "make_exception",
    "mapping": P.make_exception,
    "python_implementation": pyimpl_make_exception,
}


__primitive_defaults__ = {
    "name": "make_exception",
    "registered_name": "make_exception",
    "type": "backend",
    "python_implementation": pyimpl_make_exception,
    "inferrer_constructor": infer_make_exception,
    "grad_transform": bprop_make_exception,
}
