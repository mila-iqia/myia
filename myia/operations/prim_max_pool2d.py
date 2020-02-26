"""Definitions for the primitive `max_pool2d`."""

from .. import lib, xtype
from ..lib import SHAPE, TYPE, bprop_to_grad_transform, standard_prim
from ..operations import max_pool2d_grad, zeros_like
from . import primitives as P


@standard_prim(P.max_pool2d)
async def infer_max_pool2d(self, engine,
                           input: lib.AbstractArray,
                           kernel_size: lib.u64tup_typecheck,
                           stride: lib.u64tup_typecheck,
                           padding: lib.u64tup_typecheck,
                           dilation: lib.u64tup_typecheck,
                           ceil_mode: xtype.Bool):
    """Infer the return type of primitive `max_pool2d`."""
    # TODO: _shape_type should not allow float to be converted to uint
    # TODO: support ceil_mode == True

    assert ceil_mode.xvalue() is False

    h_in, w_in = input.xshape()[2:]

    kernel_size = tuple(self.require_constant(
                        e, argnum=f'"1:kernel_size[{edx}]"')
                        for edx, e in enumerate(kernel_size.elements))
    stride = tuple(self.require_constant(e, argnum=f'"2:stride[{edx}]"')
                   for edx, e in enumerate(stride.elements))
    padding = tuple(self.require_constant(e, argnum=f'"3:padding[{edx}]"')
                    for edx, e in enumerate(padding.elements))
    dilation = tuple(self.require_constant(e, argnum=f'"4:dilation[{edx}]"')
                     for edx, e in enumerate(dilation.elements))

    N = input.xshape()[0]
    C_out = input.xshape()[1]

    # Based on formulae in shape section of:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
    H_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
             // stride[0]) + 1
    W_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
             // stride[1]) + 1

    out_shape = (N, C_out, int(H_out), int(W_out))

    return type(input)(input.element, {SHAPE: out_shape, TYPE: input.xtype()})


@bprop_to_grad_transform(P.max_pool2d)
def bprop_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode,
                     out, dout):
    """Backpropagator for `max_pool2d`."""
    gI = max_pool2d_grad(input, kernel_size, stride, padding, dilation,
                         ceil_mode, dout)
    return (gI,
            zeros_like(kernel_size),
            zeros_like(stride),
            zeros_like(padding),
            zeros_like(dilation),
            zeros_like(ceil_mode))


__operation_defaults__ = {
    'name': 'max_pool2d',
    'registered_name': 'max_pool2d',
    'mapping': P.max_pool2d,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'max_pool2d',
    'registered_name': 'max_pool2d',
    'type': 'backend',
    'python_implementation': None,
    'inferrer_constructor': infer_max_pool2d,
    'grad_transform': bprop_max_pool2d,
}
