"""Primitive for conv_transpose2d."""

from . import primitives as P
from .. import xtype
from ..lib import (AbstractArray, ANYTHING, force_pending, SHAPE,
                   standard_prim, TYPE, u64pair_typecheck)
from typing import Optional

@standard_prim(P.conv_transpose2d)
async def infer_conv_transpose2d(
        self,
        engine,
        input: AbstractArray,
        weight: AbstractArray,
        stride: u64pair_typecheck,
        padding: u64pair_typecheck,
        output_padding: u64pair_typecheck,
        groups,
        dilation: u64pair_typecheck
):
    weight_type = await force_pending(weight.xtype())
    return AbstractArray(
        weight.element,
        {SHAPE: await force_pending(input.xshape()), TYPE: weight_type})


__operation_defaults__ = {
    'name': 'conv_transpose2d',
    'registered_name': 'conv_transpose2d',
    'mapping': P.conv_transpose2d,
    'python_implementation': None,
}

__primitive_defaults__ = {
    'name': 'conv_transpose2d',
    'registered_name': 'conv_transpose2d',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_conv_transpose2d,
    'grad_transform': None,
}
