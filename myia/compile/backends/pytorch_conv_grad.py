"""Conv2d grads from the link below.

https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
"""
import torch


# TODO: add conv1d
'''
def conv1d_input(input_size, weight, grad_output, stride, padding, dilation,
                 groups):
    r"""
    Computes gradient of conv1d with respect to input of convolution.

    This is same as the 1D transposed convolution operator under the hood but
    requires shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)

    Examples:
        >>> input = torch.randn(1,1,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    kernel_size = [weight.shape[2]]

    if input_size is None:
        raise ValueError("grad.conv1d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)

    return torch.conv_transpose1d(
        grad_output, weight, None, stride, padding, grad_input_padding, groups,
        dilation)


def conv1d_weight(input, weight_size, grad_output, stride, padding, dilation,
                  groups):
    r"""
    Computes gradient of conv1d with respect to weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
                                          the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
                                           Default: 1
        groups (int, optional): Number of blocked connections from input
                                channels to output channels. Default: 1

    Examples::
        >>> input = torch.randn(1,1,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)

    """
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2])

    grad_weight = torch.conv1d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2])

    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels, grad_weight.shape[2]).transpose(
            0, 1).narrow(2, 0, weight_size[2])
#'''


def conv2d_weight(
    input, weight_size, grad_output, stride, padding, dilation, groups
):
    r"""Computes gradient of conv2d with respect to the weight.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size: Shape of the weight gradient tensor
        grad_output: output gradient tensor
            (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
                                          the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
                                           Default: 1
        groups (int, optional): Number of blocked connections from input
                                channels to output channels. Default: 1

    Examples:
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.contiguous().repeat(
        1, in_channels // groups, 1, 1
    )
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1],
        1,
        grad_output.shape[2],
        grad_output.shape[3],
    )

    input = input.contiguous().view(
        1, input.shape[0] * input.shape[1], input.shape[2], input.shape[3]
    )

    grad_weight = torch.conv2d(
        input,
        grad_output,
        None,
        dilation,
        padding,
        stride,
        in_channels * min_batch,
    )

    grad_weight = grad_weight.contiguous().view(
        min_batch,
        grad_weight.shape[1] // min_batch,
        grad_weight.shape[2],
        grad_weight.shape[3],
    )

    if groups > 1:
        return (
            grad_weight.sum(dim=0)
            .view(
                out_channels,
                in_channels // groups,
                grad_weight.shape[2],
                grad_weight.shape[3],
            )
            .narrow(2, 0, weight_size[2])
            .narrow(3, 0, weight_size[3])
        )
    else:
        return (
            grad_weight.sum(dim=0)
            .view(
                in_channels // groups,
                out_channels,
                grad_weight.shape[2],
                grad_weight.shape[3],
            )
            .transpose(0, 1)
            .narrow(2, 0, weight_size[2])
            .narrow(3, 0, weight_size[3])
        )


# TODO: add conv3d
'''
def conv3d_input(input_size, weight, grad_output, stride, padding, dilation,
                 groups):
    r"""
    Computes gradient of conv3d with respect to input of convolution.

    This is same as the 3D transposed convolution operator under the hood but
    requires shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor
                (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : output gradient tensor
                      (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
                                          the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
                                           Default: 1
        groups (int, optional): Number of blocked connections from input
                                channels to output channels. Default: 1

    Examples:
        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    kernel_size = (weight.shape[2], weight.shape[3], weight.shape[4])

    if input_size is None:
        raise ValueError("grad.conv3d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)

    return torch.conv_transpose3d(
        grad_output, weight, None, stride, padding, grad_input_padding, groups,
        dilation)


def conv3d_weight(input, weight_size, grad_output, stride, padding, dilation,
                  groups):
    r"""
    Computes gradient of conv3d with respect to weight of the convolution.

    Args:
        input: input tensor of shape
               (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor
                      (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
                                          the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
                                           Default: 1
        groups (int, optional): Number of blocked connections from input
                                channels to output channels. Default: 1

    Examples:
        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.repeat(1, in_channels // groups, 1, 1, 1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
        grad_output.shape[3], grad_output.shape[4])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3],
                                    input.shape[4])

    grad_weight = torch.conv3d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
        grad_weight.shape[3], grad_weight.shape[4])

    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels, grad_weight.shape[2],
        grad_weight.shape[3], grad_weight.shape[4]).transpose(0, 1).narrow(
            2, 0, weight_size[2]).narrow(3, 0, weight_size[3]).narrow(
                4, 0, weight_size[4])
#'''


__all__ = ["conv2d_weight"]
