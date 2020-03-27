"""Definitions for the primitive `conv2d`."""

from .. import xtype
from ..lib import (
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractScalar,
    bprop_to_grad_transform,
    standard_prim,
    u64pair_typecheck,
)
from ..operations import (
    conv2d_grad_input,
    conv2d_weight_grad,
    shape,
    zeros_like,
)
from . import primitives as P


@standard_prim(P.conv2d)
async def infer_conv2d(
    self,
    engine,
    input: AbstractArray,
    weight: AbstractArray,
    stride: u64pair_typecheck,
    padding: u64pair_typecheck,
    dilation: u64pair_typecheck,
    groups: xtype.UInt[64],
):
    """Infer the return type of primitive `conv2d`."""
    # TODO: _shape_type should not allow float to be converted to uint
    # TODO: "groups: UInt[64]" should not allow float to be converted to uint

    h_in, w_in = input.xshape()[2:]
    kernel_size = weight.xshape()[2:]

    stride = tuple(
        self.require_constant(e, argnum=f'"2:stride[{edx}]"')
        for edx, e in enumerate(stride.elements)
    )
    padding = tuple(
        self.require_constant(e, argnum=f'"3:padding[{edx}]"')
        for edx, e in enumerate(padding.elements)
    )
    dilation = tuple(
        self.require_constant(e, argnum=f'"4:dilation[{edx}]"')
        for edx, e in enumerate(dilation.elements)
    )

    N = input.xshape()[0]
    C_out = weight.xshape()[0]

    # Based on formulae in shape section of:
    # https://pytorch.org/docs/stable/nn.html#conv2d
    H_out = (
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
        // stride[0]
    ) + 1
    W_out = (
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        // stride[1]
    ) + 1

    out_shape = (N, C_out, int(H_out), int(W_out))

    # Checks all elements of input have same xtype as all elements of weight
    engine.check(AbstractScalar, input.element, weight.element)
    # ^ TODO: PyTorch also enforces, but might want to change for mixed precis

    return type(weight)(
        weight.element, {SHAPE: out_shape, TYPE: weight.xtype()}
    )


@bprop_to_grad_transform(P.conv2d)
def bprop_conv2d(input, weight, stride, padding, dilation, groups, out, dout):
    """Backpropagator for `conv2d`."""
    gI = conv2d_grad_input(
        shape(input), weight, dout, stride, padding, dilation, groups
    )
    gW = conv2d_weight_grad(
        input, shape(weight), dout, stride, padding, dilation, groups
    )
    return (
        gI,
        gW,
        zeros_like(stride),
        zeros_like(padding),
        zeros_like(dilation),
        zeros_like(groups),
    )


__operation_defaults__ = {
    "name": "conv2d",
    "registered_name": "conv2d",
    "mapping": P.conv2d,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "conv2d",
    "registered_name": "conv2d",
    "type": "backend",
    "python_implementation": None,
    "inferrer_constructor": infer_conv2d,
    "grad_transform": bprop_conv2d,
}
