"""Implementation of gradient of conv2d wrt/ input, as a macro.

Use primitive conv_transpose2d.
"""
from ..lib import AbstractArray, AbstractTuple, Constant, macro
from . import primitives as P


def _get_int_tuple(at: AbstractTuple):
    return tuple(int(el.xvalue()) for el in at.children())


@macro
async def conv2d_grad_input(
    info,
    r_input_size,
    r_weight,
    r_grad_output,
    r_stride,
    r_padding,
    r_dilation,
    r_groups,
):
    """Return a new Apply calling conv_transpose2d with right arguments."""
    _input_size = await r_input_size.get()  # type: AbstractTuple
    _weight = await r_weight.get()  # type: AbstractArray
    _grad_output = await r_grad_output.get()  # type: AbstractArray
    _stride = await r_stride.get()  # type: AbstractTuple
    _padding = await r_padding.get()  # type: AbstractTuple
    _dilation = await r_dilation.get()  # type: AbstractTuple

    input_size = _get_int_tuple(_input_size)
    stride = _get_int_tuple(_stride)
    padding = _get_int_tuple(_padding)
    dilation = _get_int_tuple(_dilation)

    weight_shape = _weight.xshape()
    grad_output_shape = _grad_output.xshape()

    kernel_size = (weight_shape[2], weight_shape[3])

    # Compute grad input padding.

    # For a 2D convolution, tensors should have 4 dimensions.
    assert len(grad_output_shape) == 4
    assert len(input_size) == 4
    k = len(grad_output_shape) - 2
    input_size = input_size[-k:]

    min_sizes = []
    for d in range(k):
        min_sizes.append(
            (grad_output_shape[d + 2] - 1) * stride[d]
            - 2 * padding[d]
            + (kernel_size[d] - 1) * dilation[d]
            + 1
        )

    # Let's avoid checking minimum and maximum size here.
    # Backends should check it when relevant.

    grad_input_padding = tuple(input_size[d] - min_sizes[d] for d in range(k))
    # End computing.

    g = info.graph
    return g.apply(
        P.conv_transpose2d,
        r_grad_output.node,
        r_weight.node,
        r_stride.node,
        r_padding.node,
        Constant(grad_input_padding),
        r_groups.node,
        r_dilation.node,
    )


__operation_defaults__ = {
    "name": "conv2d_grad_input",
    "registered_name": "conv2d_grad_input",
    "mapping": conv2d_grad_input,
    "python_implementation": None,
}
