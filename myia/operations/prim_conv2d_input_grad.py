"""Definitions for the primitive `conv2d_input_grad`."""

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


def conv2d_input_grad():
    """Implement `conv2d_input_grad`."""


@standard_prim(P.conv2d_input_grad)
async def infer_conv2d_input_grad(self, engine,
                                  input_size: u64tup_typecheck,
                                  weight: AbstractArray,
                                  grad_output: AbstractArray,
                                  stride: u64pair_typecheck,
                                  padding: u64pair_typecheck,
                                  dilation: u64pair_typecheck,
                                  groups: xtype.UInt[64]):
    """Infer the return type of primitive `conv2d_input_grad`."""
    input_size_tuple = tuple(
        self.require_constant(i_s, argnum=0) for i_s in input_size.elements)
    return type(weight)(weight.element, {SHAPE: input_size_tuple,
                                         TYPE: weight.xtype()})


__operation_defaults__ = {
    'name': 'conv2d_input_grad',
    'registered_name': 'conv2d_input_grad',
    'mapping': P.conv2d_input_grad,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'conv2d_input_grad',
    'registered_name': 'conv2d_input_grad',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_conv2d_input_grad,
    'grad_transform': None,
}
