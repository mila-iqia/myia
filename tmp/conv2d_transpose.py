"""
Python implementation of conv2d_transpose based on Theano implementation.

Reference (2020/04/07):
https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py#L2927

"""
import warnings

import numpy as np
from scipy.signal.signaltools import _bvalfromboundary, _valfrommode
from scipy.signal.sigtools import _convolve2d
from six import integer_types


# Main function to call.
def conv2d_transpose(
    input: np.ndarray,
    filters: np.ndarray,
    output_padding=(0, 0),
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    groups=1,
):
    grad_input_op = Conv2DTranspose(
        output_padding=output_padding,
        padding=padding,
        strides=strides,
        dilation=dilation,
        groups=groups,
    )
    return grad_input_op.run(filters, input)


class Conv2DTranspose:

    __slots__ = (
        "convdim",
        "output_padding",
        "padding",
        "strides",
        "dilation",
        "groups",
    )

    def __init__(
        self,
        output_padding=(0, 0),
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        groups=1,
    ):
        self.convdim = 2
        self.output_padding = tuple(output_padding)
        self.padding = tuple(padding)
        self.strides = tuple(strides)
        self.dilation = tuple(dilation)
        self.groups = groups

    def run(self, kern: np.ndarray, topgrad: np.ndarray):
        new_shape = (topgrad.shape[0], topgrad.shape[1]) + tuple(
            (topgrad.shape[-self.convdim + i] - 1) * self.strides[i]
            + self.output_padding[i]
            + 1
            for i in range(self.convdim)
        )
        new_topgrad = np.zeros((new_shape), dtype=topgrad.dtype)
        new_topgrad[
            (slice(None), slice(None))
            + tuple(
                slice(None, None, self.strides[i]) for i in range(self.convdim)
            )
        ] = topgrad
        topgrad = new_topgrad

        kern = self.correct_for_groups(kern)

        # "fill" padding
        conv_padding = tuple(
            (kern.shape[-self.convdim + i] - 1) * self.dilation[i]
            for i in range(self.convdim)
        )

        img = conv(
            topgrad,
            kern,
            padding=conv_padding,
            dilation=self.dilation,
            groups=self.groups,
            conv_mode="full",
        )

        if any(p != 0 for p in self.padding):
            img = img[
                (slice(None), slice(None))
                + tuple(
                    slice(self.padding[i], img.shape[i + 2] - self.padding[i])
                    for i in range(self.convdim)
                )
            ]

        return img

    def correct_for_groups(self, mat):
        mshp0 = mat.shape[0] // self.groups
        mshp1 = mat.shape[-self.convdim - 1] * self.groups
        mat = mat.reshape((self.groups, mshp0) + mat.shape[1:])
        mat = mat.transpose((1, 0, 2) + tuple(range(3, 3 + self.convdim)))
        mat = mat.reshape((mshp0, mshp1) + mat.shape[-self.convdim :])
        mat = mat.transpose((1, 0) + tuple(range(2, 2 + self.convdim)))
        return mat


def conv(
    img, kern, padding=(0, 0), dilation=(1, 1), groups=1, conv_mode="valid"
):
    """Basic slow Python 2D convolution for DebugMode"""
    convdim = 2
    if isinstance(dilation, integer_types):
        dilation = (dilation,) * convdim
    if len(dilation) != convdim:
        raise ValueError(
            "invalid dilation {}, expected {} values".format(dilation, convdim)
        )

    out_shape = get_conv_output_shape(
        img.shape, kern.shape, padding, [1] * convdim, dilation
    )

    dil_kern_shp = kern.shape[:-convdim] + tuple(
        (kern.shape[-convdim + i] - 1) * dilation[i] + 1 for i in range(convdim)
    )
    dilated_kern = np.zeros(dil_kern_shp, dtype=kern.dtype)
    dilated_kern[
        (slice(None),) * (dilated_kern.ndim - convdim)
        + tuple(slice(None, None, dilation[i]) for i in range(convdim))
    ] = kern
    out = np.zeros(out_shape, dtype=img.dtype)

    if img.shape[1] % groups != 0:
        raise ValueError(
            "number of input channels must be divible by num_groups"
        )
    if kern.shape[0] % groups != 0:
        raise ValueError("number of filters must be divisible by num_groups")
    if img.shape[1] // groups != kern.shape[1]:
        raise ValueError(
            "the number of input channels in the kernel should "
            "specify the number of channels of 1 group"
        )
    input_channel_offset = img.shape[1] // groups
    output_channel_offset = kern.shape[0] // groups

    val = _valfrommode(conv_mode)
    bval = _bvalfromboundary("fill")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.ComplexWarning)
        for b in range(img.shape[0]):
            for g in range(groups):
                for n in range(output_channel_offset):
                    for im0 in range(input_channel_offset):
                        # some cast generates a warning here
                        out[
                            b, g * output_channel_offset + n, ...
                        ] += _convolve2d(
                            img[b, g * input_channel_offset + im0, ...],
                            dilated_kern[
                                g * output_channel_offset + n, im0, ...
                            ],
                            1,
                            val,
                            bval,
                            0,
                        )

    return out


def get_conv_output_shape(
    image_shape, kernel_shape, border_mode, subsample, filter_dilation=None
):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: tuple of int (symbolic or numeric) corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. For a normal convolution, its four (for 2D convolution)
        or five (for 3D convolution) elements must correspond respectively to :
        number of output channels, number of input channels, height and width
        (and possibly depth) of the kernel.
        None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric) or pairs of ints. If it is a string, it must be 'valid',
        'half' or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis. For asymmetric padding, provide a pair of ints for each dimension.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        espectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and width axis.
    Note - The shape of the convolution output does not depend on  the 'num_groups' parameters.

    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image. None where undefined.

    """
    bsize, imshp = image_shape[0], image_shape[2:]

    convdim = len(image_shape) - 2
    nkern, kshp = kernel_shape[0], kernel_shape[-convdim:]

    if filter_dilation is None:
        filter_dilation = np.ones(len(subsample), dtype="int")

    if isinstance(border_mode, tuple):
        out_shp = tuple(
            get_conv_shape_1axis(
                imshp[i],
                kshp[i],
                border_mode[i],
                subsample[i],
                filter_dilation[i],
            )
            for i in range(len(subsample))
        )
    else:
        out_shp = tuple(
            get_conv_shape_1axis(
                imshp[i], kshp[i], border_mode, subsample[i], filter_dilation[i]
            )
            for i in range(len(subsample))
        )
    return (bsize, nkern) + out_shp


def get_conv_shape_1axis(
    image_shape, kernel_shape, border_mode, subsample, dilation=1
):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    border_mode: string, int or tuple of 2 ints. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis. If it is a tuple, its two elements
        must correspond to the asymmetric padding (e.g., left and right) on
        the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis. None if undefined.

    """
    if None in [image_shape, kernel_shape, border_mode, subsample, dilation]:
        return None
    # Implicit dilated kernel shape
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "half":
        pad_l = pad_r = dil_kernel_shape // 2
    elif border_mode == "full":
        pad_l = pad_r = dil_kernel_shape - 1
    elif border_mode == "valid":
        pad_l = pad_r = 0
    else:
        if isinstance(border_mode, tuple):
            pad_l, pad_r = border_mode
        else:
            pad_l = pad_r = border_mode
        if pad_l < 0 or pad_r < 0:
            raise ValueError("border_mode must be >= 0")

    # In case of symbolic shape, we want to build the smallest graph
    # (image_shape + 2 * pad - dil_kernel_shape) // subsample + 1
    out_shp = image_shape - dil_kernel_shape
    if pad_l != 0:
        out_shp += pad_l
    if pad_r != 0:
        out_shp += pad_r
    if subsample != 1:
        out_shp = out_shp // subsample
    out_shp = out_shp + 1

    return out_shp
