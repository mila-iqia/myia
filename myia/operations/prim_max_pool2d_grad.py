"""Definitions for the primitive `max_pool2d_grad`."""

from .. import lib, xtype
from ..lib import standard_prim
from . import primitives as P


@standard_prim(P.max_pool2d_grad)
async def infer_max_pool2d_grad(
    self,
    engine,
    input: lib.AbstractArray,
    kernel_size: lib.u64tup_typecheck,
    stride: lib.u64tup_typecheck,
    padding: lib.u64tup_typecheck,
    dilation: lib.u64tup_typecheck,
    ceil_mode: xtype.Bool,
    dout: lib.AbstractArray,
):
    """Infer the return type of primitive `max_pool2d_grad`."""
    return input


__operation_defaults__ = {
    "name": "max_pool2d_grad",
    "registered_name": "max_pool2d_grad",
    "mapping": P.max_pool2d_grad,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "max_pool2d_grad",
    "registered_name": "max_pool2d_grad",
    "type": "backend",
    "python_implementation": None,
    "inferrer_constructor": infer_max_pool2d_grad,
    "grad_transform": None,
}
