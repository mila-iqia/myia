"""Implementation of complext primitives."""
import itertools
import warnings

import numpy as np
from scipy.signal.signaltools import _bvalfromboundary, _valfrommode
from scipy.signal.sigtools import _convolve2d


def _get_conv_shape_1axis(
    image_shape, kernel_shape, border_mode, subsample, dilation=1
):
    """This function compute the output shape of convolution operation.

    Copied and simplified from theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py

    Parameters
    ----------
    image_shape: int
        Corresponds to the input image shape on a given axis.
    kernel_shape: int
        Corresponds to the kernel shape on a given axis.
    border_mode: string or int. If it is a string, it must be
        'valid' or 'full'.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis.

    """
    # Implicit dilated kernel shape
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "full":
        pad_l = pad_r = dil_kernel_shape - 1
    elif border_mode == "valid":
        pad_l = pad_r = 0
    else:
        assert border_mode >= 0
        pad_l = pad_r = border_mode

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


def _get_conv_output_shape(
    image_shape, kernel_shape, border_mode, subsample, filter_dilation
):
    """This function compute the output shape of convolution operation.

    Copied and simplified from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py

    Parameters
    ----------
    image_shape: tuple of int corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int corresponding to the
        kernel shape. For a normal convolution, its four (for 2D convolution)
        or five (for 3D convolution) elements must correspond respectively to :
        number of output channels, number of input channels, height and width
        (and possibly depth) of the kernel.
        For an unshared 2D convolution, its six channels must correspond to :
        number of output channels, height and width of the output, number of
        input channels, height and width of the kernel.
        None where undefined.
    border_mode: string, or tuple of int. If it is a string, it must be 'valid'
        or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis.
    subsample: tuple of int. Its two or three elements
        respectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int. Its two or three
        elements correspond respectively to the dilation on height and width axis.

    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image.

    """
    bsize, imshp = image_shape[0], image_shape[2:]

    convdim = len(image_shape) - 2
    nkern, kshp = kernel_shape[0], kernel_shape[-convdim:]

    if isinstance(border_mode, tuple):
        out_shp = tuple(
            _get_conv_shape_1axis(
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
            _get_conv_shape_1axis(
                imshp[i], kshp[i], border_mode, subsample[i], filter_dilation[i]
            )
            for i in range(len(subsample))
        )
    return (bsize, nkern) + out_shp


def _conv2d(img, kern, mode="valid", dilation=(1, 1), groups=1):
    """Basic slow Python 2D or 3D convolution for DebugMode.

    Copied and simplified from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py
    """
    convdim = 2
    assert mode in ("valid", "full")
    out_shape = _get_conv_output_shape(
        img.shape, kern.shape, mode, [1] * convdim, dilation
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

    input_channel_offset = img.shape[1] // groups
    output_channel_offset = kern.shape[0] // groups

    val = _valfrommode(mode)
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


def conv2d(
    inp, weight, strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    """Publc implementation of conv2d.

    Copied and simplified from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py
    """

    convdim = 2
    assert groups > 0
    assert len(strides) == len(dilation) == len(padding) == convdim
    assert all(mode >= 0 for mode in padding)

    pad = tuple((m, m) for m in padding)

    if any(p != (0, 0) for p in pad):
        new_img = np.zeros(
            (inp.shape[0], inp.shape[1])
            + tuple(
                inp.shape[i + 2] + pad[i][0] + pad[i][1] for i in range(convdim)
            ),
            dtype=inp.dtype,
        )
        new_img[
            (slice(None), slice(None))
            + tuple(
                slice(pad[i][0], inp.shape[i + 2] + pad[i][0])
                for i in range(convdim)
            )
        ] = inp
        inp = new_img

    conv_out = _conv2d(
        inp,
        weight[:, :, ::-1, ::-1],
        mode="valid",
        dilation=dilation,
        groups=groups,
    )
    return conv_out[
        (slice(None), slice(None))
        + tuple(slice(None, None, strides[i]) for i in range(convdim))
    ]


def conv2d_weight_grad(
    input, weight_size, grad_output, stride, padding, dilation, groups
):
    """Computes gradient of conv2d with respect to the weight.

    Adapted from Pytorch backend.
    """
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = np.tile(
        np.ascontiguousarray(grad_output), (1, in_channels // groups, 1, 1)
    )
    grad_output = np.ascontiguousarray(grad_output).reshape(
        (
            grad_output.shape[0] * grad_output.shape[1],
            1,
            grad_output.shape[2],
            grad_output.shape[3],
        )
    )

    input = np.ascontiguousarray(input).reshape(
        (1, input.shape[0] * input.shape[1], input.shape[2], input.shape[3])
    )

    grad_weight = conv2d(
        inp=input,
        weight=grad_output,
        dilation=stride,
        padding=padding,
        strides=dilation,
        groups=in_channels * min_batch,
    )

    grad_weight = np.ascontiguousarray(grad_weight).reshape(
        (
            min_batch,
            grad_weight.shape[1] // min_batch,
            grad_weight.shape[2],
            grad_weight.shape[3],
        )
    )

    if groups > 1:
        return np.sum(grad_weight, axis=0).reshape(
            (
                out_channels,
                in_channels // groups,
                grad_weight.shape[2],
                grad_weight.shape[3],
            )
        )[:, :, : weight_size[2], :][:, :, :, : weight_size[3]]
    else:
        return (
            np.sum(grad_weight, axis=0)
            .reshape(
                (
                    in_channels // groups,
                    out_channels,
                    grad_weight.shape[2],
                    grad_weight.shape[3],
                )
            )
            .transpose(1, 0, 2, 3)[:, :, : weight_size[2], :][
                :, :, :, : weight_size[3]
            ]
        )


def conv_transpose2d(
    data, weight, strides, padding, output_padding, groups, dilation
):
    """Implement conv2d_transpose using conv2d.

    Adapted from Theano and Relay backend.

    Theano reference (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py
    """
    data_shape = data.shape
    kern_shape = weight.shape
    n, _, h_in, w_in = data_shape
    filter_h, filter_w = kern_shape[2:]
    c_out = kern_shape[1] * groups

    h_out = (
        (h_in - 1) * strides[0]
        - 2 * padding[0]
        + dilation[0] * (filter_h - 1)
        + output_padding[0]
        + 1
    )
    w_out = (
        (w_in - 1) * strides[1]
        - 2 * padding[1]
        + dilation[1] * (filter_w - 1)
        + output_padding[1]
        + 1
    )

    kern = weight
    topgrad = data
    shape = (h_out, w_out)
    imshp = (n, c_out, h_out, w_out)
    convdim = 2

    assert topgrad.ndim == kern.ndim == 2 + convdim

    dil_kernshp = tuple(
        (kern.shape[-convdim + i] - 1) * dilation[i] + 1 for i in range(convdim)
    )

    pad = tuple((m, m) for m in padding)

    expected_topgrad_shape = _get_conv_output_shape(
        imshp, kern.shape, padding, strides, dilation
    )
    if expected_topgrad_shape != tuple(topgrad.shape):
        # If expected topgrad is larger than given topgrad,
        # padding dimensions at end seems sufficient to produce
        # right conv_transpose2d output.
        assert all(
            expected >= given
            for (expected, given) in zip(expected_topgrad_shape, topgrad.shape)
        ), (
            "invalid input_shape for gradInputs: the given input_shape "
            "would produce an output of shape {}, but the given topgrad "
            "has shape {}".format(
                tuple(expected_topgrad_shape), tuple(topgrad.shape)
            )
        )
        tmp = np.zeros(expected_topgrad_shape, dtype=topgrad.dtype)
        tmp[tuple(slice(None, val) for val in topgrad.shape)] = topgrad
        topgrad = tmp

    if any(strides[i] > 1 for i in range(convdim)):
        new_shape = (topgrad.shape[0], topgrad.shape[1]) + tuple(
            shape[i] + pad[i][0] + pad[i][1] - dil_kernshp[i] + 1
            for i in range(convdim)
        )
        new_topgrad = np.zeros(new_shape, dtype=topgrad.dtype)
        new_topgrad[
            (slice(None), slice(None))
            + tuple(slice(None, None, strides[i]) for i in range(convdim))
        ] = topgrad
        topgrad = new_topgrad

    def correct_for_groups(mat):
        mshp0 = mat.shape[0] // groups
        mshp1 = mat.shape[-convdim - 1] * groups
        mat = mat.reshape((groups, mshp0) + mat.shape[1:])
        mat = mat.transpose((1, 0, 2) + tuple(range(3, 3 + convdim)))
        mat = mat.reshape((mshp0, mshp1) + mat.shape[-convdim:])
        return mat

    kern = correct_for_groups(kern)

    axes_order = (1, 0) + tuple(range(2, 2 + convdim))
    kern = kern.transpose(axes_order)
    img = _conv2d(topgrad, kern, mode="full", dilation=dilation, groups=groups)

    if any(p != (0, 0) for p in pad):
        img = img[
            (slice(None), slice(None))
            + tuple(
                slice(pad[i][0], img.shape[i + 2] - pad[i][1])
                for i in range(convdim)
            )
        ]

    return img


def array_reduce(fn, array, shp):
    """Implement `array_reduce`.

    Copied from primitive pyimpl.
    """
    idtype = array.dtype
    ufn = np.frompyfunc(fn, 2, 1)
    delta = len(array.shape) - len(shp)
    assert delta >= 0, "Shape to reduce to cannot be larger than original"

    def is_reduction(ishp, tshp):
        if tshp == 1 and ishp > 1:
            return True
        else:
            assert tshp == ishp, "Dimension mismatch for reduce"
            return False

    reduction = [
        (delta + idx if is_reduction(ishp, tshp) else None, True)
        for idx, (ishp, tshp) in enumerate(zip(array.shape[delta:], shp))
    ]

    reduction = [(i, False) for i in range(delta)] + reduction

    for idx, keep in reversed(reduction):
        if idx is not None:
            array = ufn.reduce(array, axis=idx, keepdims=keep)

    if not isinstance(array, np.ndarray):
        # Force result to be ndarray, even if it's 0d
        array = np.array(array)

    array = array.astype(idtype)

    return array


def take_grad_inp(nb_indices, indices, values):
    """Implementation for primitive `take_grad_inp`."""
    row_size = values.shape[-1]
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = np.zeros((nb_indices, row_size), dtype=values.dtype)
    for i in range(nb_indices):
        output[i] = (
            ((broadcastable_indices == i) * values)
            .reshape((-1, row_size))
            .sum(axis=0)
        )
    return output


def scatter(x, axis, indices, src):
    """Implementation of scatter primitive."""
    axis = axis if axis >= 0 else axis + len(x.shape)
    assert axis >= 0
    assert axis < len(x.shape)
    output = x.copy()
    for index in np.ndindex(*indices.shape):
        new_index = list(index)
        new_index[axis] = indices[index]
        output[tuple(new_index)] = src[index]
    return output


def scatter_add(x, axis, indices, src):
    """Implementation of scatter_add primitive."""
    axis = axis if axis >= 0 else axis + len(x.shape)
    assert axis >= 0
    assert axis < len(x.shape)
    output = x.copy()
    for index in np.ndindex(*indices.shape):
        new_index = list(index)
        new_index[axis] = indices[index]
        output[tuple(new_index)] += src[index]
    return output


def argmax(x, dim):
    """Implementation of argmax primitive.

    Adapted from Pytorch backend.
    """
    dim = tuple(sorted(dim))
    n = ()
    for _s in range(len(x.shape)):
        if _s not in dim:
            n = n + (_s,)
    n = n + dim
    # x = x.permute(n)
    x = np.transpose(x, n)
    ns = x.shape[0 : -len(dim)] + (-1,)
    r = np.argmax(x.reshape(ns), -1)
    rl = list(r.shape)
    for _sd in dim:
        rl.insert(_sd, 1)
    rf = tuple(rl)
    return np.reshape(r, rf)


def _max_pool2d_out_shape(
    imgshape, ws, stride, pad, ndim,
):
    """Return the shape of max_pool2d output.

    Adapted from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/signal/pool.py

    Parameters
    ----------
    imgshape : tuple, list, or similar of integer or scalar Theano variable
        The shape of a tensor of images. The last N elements are
        interpreted as the number of rows, and the number of cols.
    ws : list or tuple of N ints
        Downsample factor over rows and column.
        ws indicates the pool region size.
    stride : list or tuple of N ints
        Stride size, which is the number of shifts over rows/cols/slices to get the
        next pool region.
    pad : tuple of N ints
        For each downsampling dimension, this specifies the number of zeros to
        add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
        size of the top and bottom margins, pad_w specifies the size of the left and
        right margins.
    ndim : int
        The number of pooling dimensions N.
        The default is 2.

    Returns
    -------
    list
        The shape of the output from this op, for input of given shape.
    """

    assert ndim > 0
    assert (
        len(imgshape) >= ndim
    ), "imgshape must have at least {} dimensions".format(ndim)

    # Compute output shape based on formula on Torch page (2020/11/16):
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    h_in, w_in = imgshape[-ndim:]
    h_out = int((h_in + 2 * pad[0] - (ws[0] - 1) - 1) // stride[0] + 1)
    w_out = int((w_in + 2 * pad[1] - (ws[1] - 1) - 1) // stride[1] + 1)

    rval = list(imgshape[:-ndim]) + [h_out, w_out]
    return rval


def max_pool2d(x, ws, stride, pad, ceil_mode):
    """Implementation of max_pool2d.

    Adapted from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/signal/pool.py
    """

    nd = 2
    inc_pad = False

    assert len(ws) == len(stride) == len(pad) == nd
    assert ceil_mode is False

    if len(x.shape) < nd:
        raise NotImplementedError(
            "Pool requires input with {} or more dimensions".format(nd)
        )

    z_shape = _max_pool2d_out_shape(x.shape, ws, stride, pad, nd)
    assert all(z > 0 for z in z_shape[-nd:])
    zz = np.empty(z_shape, dtype=x.dtype)
    # size of pooling output
    pool_out_shp = zz.shape[-nd:]
    img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in range(nd))

    # pad the image
    if max(pad) != 0:
        y = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
        y[
            (slice(None),) * (len(x.shape) - nd)
            + tuple(slice(pad[i], img_shp[i] - pad[i]) for i in range(nd))
        ] = x
    else:
        y = x
    func = np.max

    # precompute the region boundaries for each dimension
    region_slices = [[] for i in range(nd)]
    for i in range(nd):
        for j in range(pool_out_shp[i]):
            start = j * stride[i]
            end = min(start + ws[i], img_shp[i])
            if not inc_pad:
                start = max(start, pad[i])
                end = min(end, img_shp[i] - pad[i])
            region_slices[i].append(slice(start, end))

    # iterate over non-pooling dimensions
    for k in np.ndindex(*x.shape[:-nd]):
        zzk = zz[k]
        yk = y[k]
        # iterate over pooling regions
        for r in np.ndindex(*pool_out_shp):
            reg = yk[[region_slices[i][r[i]] for i in range(nd)]]
            # reg may have a dimension with size 0.
            zzk[r] = func(reg) if reg.size else 0

    return zz


def max_pool2d_grad(x, ws, stride, pad, ceil_mode, gz):
    """Implementation of max_pool2d_grad.

    Adapted from Theano (2020/11/08):
    https://github.com/Theano/Theano/blob/master/theano/tensor/signal/pool.py
    """

    maxout = max_pool2d(x, ws, stride, pad, ceil_mode)
    nd = 2
    assert len(ws) == len(stride) == len(pad) == nd
    if len(x.shape) < nd:
        raise NotImplementedError(
            "MaxPoolGrad requires input with {} or more dimensions".format(nd)
        )
    pool_out_shp = maxout.shape[-nd:]
    img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in range(nd))

    # pad the image
    if max(pad) != 0:
        y = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
        y[
            (slice(None),) * (len(x.shape) - nd)
            + tuple(slice(pad[i], img_shp[i] - pad[i]) for i in range(nd))
        ] = x
    else:
        y = x
    gx = np.zeros_like(y)

    # precompute the region boundaries for each dimension
    region_ranges = [[] for i in range(nd)]
    for i in range(nd):
        for j in range(pool_out_shp[i]):
            start = max(j * stride[i], pad[i])
            end = min(start + ws[i], img_shp[i])
            region_ranges[i].append(range(start, end))

    # iterate over non-pooling dimensions
    for k in np.ndindex(*x.shape[:-nd]):
        gxk = gx[k]
        gzk = gz[k]
        yk = y[k]
        maxoutk = maxout[k]
        # iterate over pooling regions
        for r in np.ndindex(*pool_out_shp):
            maxout_value = maxoutk[r]
            # iterate inside region
            for c in itertools.product(
                *[region_ranges[i][r[i]] for i in range(nd)]
            ):
                if maxout_value == yk[c]:
                    gxk[c] += gzk[r]

    # unpad the image
    gx = gx[
        (slice(None),) * (len(x.shape) - nd)
        + tuple(slice(pad[i], img_shp[i] - pad[i]) for i in range(nd))
    ]
    return gx
