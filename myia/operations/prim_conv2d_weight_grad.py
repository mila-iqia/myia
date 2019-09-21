"""Definitions for the primitive `conv2d_weight_grad`."""

from .. import xtype
from ..lib import (
    SHAPE,
    TYPE,
    AbstractArray,
    standard_prim,
    u64pair_typecheck,
    u64tup_typecheck,
)
from . import primitives as P


@standard_prim(P.conv2d_weight_grad)
async def infer_conv2d_weight_grad(self, engine, input: AbstractArray,
                                   weight_size: u64tup_typecheck,
                                   grad_output: AbstractArray,
                                   stride: u64pair_typecheck,
                                   padding: u64pair_typecheck,
                                   dilation: u64pair_typecheck,
                                   groups: xtype.UInt[64]):
    """Infer the return type of primitive `conv2d_weight_grad`."""
    weight_size_tuple = tuple(
        self.require_constant(w_s, argnum=0) for w_s in weight_size.elements)
    return type(input)(input.element, {SHAPE: weight_size_tuple,
                                       TYPE: input.xtype()})


__operation_defaults__ = {
    'name': 'conv2d_weight_grad',
    'registered_name': 'conv2d_weight_grad',
    'mapping': P.conv2d_weight_grad,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'conv2d_weight_grad',
    'registered_name': 'conv2d_weight_grad',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_conv2d_weight_grad,
    'grad_transform': None,
}
