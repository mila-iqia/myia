"""Definitions for the primitive `array_cast`."""

from .. import lib, operations, xtype
from ..lib import (
    MyiaTypeError,
    bprop_to_grad_transform,
    standard_prim,
    type_to_abstract,
)
from . import primitives as P


def pyimpl_array_cast(x, t):
    """Implement `array_cast`."""
    t = type_to_abstract(t)
    assert isinstance(t, lib.AbstractScalar)
    t = t.values[lib.TYPE]
    assert issubclass(t, xtype.Number)
    dtype = xtype.type_to_np_dtype(t)
    return x.astype(dtype)
    # TODO


@standard_prim(P.array_cast)
async def infer_array_cast(
    self, engine, a: lib.AbstractArray, typ: lib.AbstractType
):
    """Infer the return type of primitive `array_cast`."""
    scal = typ.element
    if not isinstance(scal, lib.AbstractScalar):
        raise MyiaTypeError("array_cast must cast to a scalar dtype")
    t = scal.xtype()
    engine.check(xtype.Number, t)
    e_values = {**a.element.values, lib.TYPE: t}
    return lib.AbstractArray(lib.AbstractScalar(e_values), a.values)


# TODO: Need dtype attr implementation for xtype.NDArray to get dtype of x.
#       dtype attr/method will go in xtype.NDArray dict in standard_method_map
#       in myia.pipeline.resources
@bprop_to_grad_transform(P.array_cast)
def bprop_array_cast(x, t, out, dout):
    """Backpropagator for primitive `array_cast`."""
    return (operations.array_cast(dout, operations.dtype(x)), t)


__operation_defaults__ = {
    "name": "array_cast",
    "registered_name": "array_cast",
    "mapping": P.array_cast,
    "python_implementation": pyimpl_array_cast,
}


__primitive_defaults__ = {
    "name": "array_cast",
    "registered_name": "array_cast",
    "type": "backend",
    "python_implementation": pyimpl_array_cast,
    "inferrer_constructor": infer_array_cast,
    "grad_transform": bprop_array_cast,
}
