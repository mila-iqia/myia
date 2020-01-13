"""Primitive for conv_transpose2d."""

from . import primitives as P
from ..lib import (AbstractArray, AbstractScalar, AbstractTuple, force_pending,
                   SHAPE, standard_prim, TYPE)


@standard_prim(P.conv_transpose2d)
async def infer_conv_transpose2d(
        self,
        engine,
        input: AbstractArray,
        weight: AbstractArray,
        stride: AbstractTuple,
        padding: AbstractTuple,
        output_padding: AbstractTuple,
        groups: AbstractScalar,
        dilation: AbstractTuple
):
    n, c_in, h_in, w_in = await force_pending(input.xshape())
    stride = tuple(
        self.require_constant(e, argnum=f'"2:stride[{edx}]"')
        for edx, e in enumerate(stride.elements))
    padding = tuple(
        self.require_constant(e, argnum=f'"3:padding[{edx}]"')
        for edx, e in enumerate(padding.elements))
    output_padding = tuple(
        self.require_constant(e, argnum=f'"4:output_padding[{edx}]"')
        for edx, e in enumerate(output_padding.elements))
    groups = self.require_constant(groups, argnum='5:groups')
    dilation = tuple(
        self.require_constant(e, argnum=f'"6:dilation[{edx}]"')
        for edx, e in enumerate(dilation.elements))

    _, c_out_per_group, kh, kw = await force_pending(weight.xshape())
    c_out = c_out_per_group * groups
    h_out = int(
        (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1)
        + output_padding[0] + 1
    )
    w_out = int(
        (w_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1)
        + output_padding[1] + 1
    )
    output_shape = n, c_out, h_out, w_out
    weight_type = await force_pending(weight.xtype())
    return AbstractArray(
        weight.element, {SHAPE: output_shape, TYPE: weight_type})


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
