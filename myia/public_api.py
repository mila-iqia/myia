"""Myia Frontend API of functions."""

#############################################################################
# WARNING:  None of this file is explicitly executed by pytest or forced    #
#           to be covered by Codecov. It's omitted in .coveragerc           #
#                                                                           #
#           It is instead only parsed by Myia.                              #
#                                                                           #
#           Output & Gradients of each function in this is/needs_to_be      #
#           compared with original pytorch function that function is        #
#           replacing via pytorch_*_map in /myia/frontends/pytorch.py       #
#                                                                           #
#           I.e. Every function in this should have a pytest test made      #
#           for it that asserts its Output & Gradients is equal to the      #
#           pytorch original.                                               #
#############################################################################

import operator
from functools import reduce

import numpy as np

from . import operations
from .abstract import myia_static
from .frontends.abstract_types import AA, AS, AA_bool
from .hypermap import hyper_map
from .operations import primitives as P
from .utils import MyiaValueError, core
from .xtype import TupleT, f32, i64, u64

# ############# THESE FUNCTIONS SHOULD BE IN ALPHABETICAL ORDER #############


# HELPER FUNCTIONS ##########################################################


@myia_static
def _chunks_to_split_sections(a_dim_shp, chunks):
    rem = a_dim_shp % chunks
    sections = ()
    if rem == 0:
        def_sec_size = int(a_dim_shp / chunks)
        for i in range(chunks):
            sections = sections + (def_sec_size,)
    elif a_dim_shp < chunks:
        for i in range(a_dim_shp):
            sections = sections + (1,)
    else:
        def_sec_size = a_dim_shp // chunks + 1
        sec_rem = (a_dim_shp // def_sec_size) % chunks
        for i in range(chunks - sec_rem):
            sections = sections + (def_sec_size,)
        new_rem = a_dim_shp % (def_sec_size * (chunks - sec_rem))
        if new_rem != 0 and (chunks < a_dim_shp):
            sections = sections + (new_rem,)
    return sections


@myia_static
def _dim_explicit(a_shp, dim):
    if dim is None:
        return dim

    if dim < 0:
        dim = len(a_shp) + dim
    return dim


@myia_static
def _dim_explicit_unsqueeze(a_shp, dim):
    if dim < 0:
        dim = len(a_shp) + dim + 1
    return dim


@myia_static
def _dim_tuple_explicit(a_shp, dim):
    shp_explicit = ()
    for s in dim:
        if s < 0:
            shp_explicit = shp_explicit + (len(a_shp) + s,)
        else:
            shp_explicit = shp_explicit + (s,)
    return shp_explicit


@myia_static
def _build_fwd_tuple(shp):
    t = ()
    for d in range(len(shp)):
        t = t + (d,)
    return t


@core
def _ensure_u64(x):
    assert P.hastype(x, i64) or P.hastype(x, u64)
    assert x >= 0
    return P.scalar_cast(x, u64)


@core
def _pair(x):
    if not P.hastype(x, TupleT):
        x = (x, x)
    x = (_ensure_u64(x[0]), x[1])
    x = (x[0], _ensure_u64(x[1]))
    return x


@myia_static
def _prepare_dims_to_norm(shape, dim):
    # Make sure dim is a tuple.
    if not isinstance(dim, tuple):
        dim = (dim,)
    # If no dim, norm all dimensions.
    if len(dim) == 0:
        dim = shape

    leading_dims = ()
    dims_to_norm = ()
    for i in range(len(shape)):
        if i in dim:
            dims_to_norm = dims_to_norm + (i,)
        else:
            leading_dims = leading_dims + (i,)
    permutation = ()
    leading_shape = ()
    reshaping_tuple = ()
    if leading_dims:
        permutation = leading_dims + dims_to_norm
        leading_shape = tuple(shape[d] for d in leading_dims)
        reshaping_tuple = leading_shape + (-1,)
    else:
        leading_dims = None
    return leading_dims, permutation, leading_shape, reshaping_tuple


def prod(x):
    return reduce(operator.mul, x, 1)


@myia_static
def _shp_explicit(a_shp, shp):

    unk_count = 0
    for s in shp:
        if s < -1:
            e_msg = "New shape cannot contain value less than -1 in reshape"
            raise MyiaValueError(e_msg)
        if s == -1:
            unk_count = unk_count + 1

    if unk_count > 1:
        e_msg = "New shape can only contain 1 unknown (-1) dim in reshape"
        raise MyiaValueError(e_msg)

    if (
        prod(a_shp) % prod(shp) != 0
        if unk_count == 1
        else prod(shp) != prod(a_shp)
    ):
        e_msg = "Cannot change the total number of elements in reshape"
        raise MyiaValueError(e_msg)

    known_unk_dim = int(abs(prod(a_shp) / prod(shp)))
    shp_explicit = ()
    for s in shp:
        if s == -1:
            shp_explicit = shp_explicit + (known_unk_dim,)
        else:
            shp_explicit = shp_explicit + (s,)

    return shp_explicit


@myia_static
def _shp_squeeze(orig_shp, dim, keepdim):

    final_shape = ()
    skip = False

    if keepdim:
        final_shape = orig_shp
    else:
        if dim is not None:
            if orig_shp[dim] != 1:
                final_shape = orig_shp
                skip = True
        if not skip:
            new_shape = ()
            if dim is None:
                for _x in orig_shp:
                    if _x != 1:
                        new_shape = new_shape + (_x,)
            else:
                i = 0
                for _x in orig_shp:
                    if _x == 1 and dim == i:
                        new_shape = new_shape
                    else:
                        new_shape = new_shape + (_x,)
                    i = i + 1
            final_shape = new_shape
    return final_shape


@myia_static
def _shp_unsqueeze(orig_shp, dim):

    final_shape = ()

    for ddx, d in enumerate(orig_shp):
        if ddx == dim:
            final_shape = final_shape + (1, d)
        else:
            final_shape = final_shape + (d,)

    if dim == len(orig_shp):
        final_shape = final_shape + (1,)

    return final_shape


@myia_static
def _total_elements(shp):

    _tot_elems = int(prod(shp))

    return _tot_elems


@myia_static
def _var_denom(shp, dim, unbiased):

    if dim is None:
        denom = int(prod(shp))
    else:
        denom = shp[dim]

    if unbiased is True:
        denom = denom - 1

    return denom


# ACTUAL FUNCTIONS ##########################################################


@core
def argmax(self, dim=None, keepdim=False):
    """Map of 'argmax' pytorch method."""
    x = self
    dim_orig = dim
    if dim is None:
        dim = _build_fwd_tuple(x.shape)
    elif isinstance(dim, int):
        dim = (dim,)
    dim = _dim_tuple_explicit(x.shape, dim)
    ret = P.argmax(x, dim)
    final_shape = _shp_squeeze(
        ret.shape, _dim_explicit(x.shape, dim_orig), keepdim
    )
    return P.reshape(ret, final_shape)


@core
def binary_cross_entropy(input, target, reduction="mean"):
    """Map of method torch.nn.functional.binary_cross_entropy."""
    out = -(operations.array_log(input) * target + (1. - target) * operations.array_log(1. - input))
    if reduction == "none":
        out = out
    elif reduction == "mean":
        out = mean(out)
    elif reduction == "sum":
        out = _sum(out)
    return out


@core
def cat(self, dim=0):
    """Map of 'cat' pytorch method."""
    x = self
    dim = _dim_explicit(x[0].shape, dim)
    return P.concat(x, dim)


@core
def chunk(self, chunks, dim=0):
    """Map of 'chunk' pytorch method."""
    sections = _chunks_to_split_sections(self.shape[dim], chunks)
    return P.split(self, sections, dim)


@core
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    r"""Applies a Conv2d."""
    # noqa: D202
    """
    # This is for later versions of pytorch that support other paddings?
    if padding_mode != 'zeros':
        raise Exception("'zeros' is the only padding_mode that is currently
                        supported.")
    #"""

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    groups = _ensure_u64(groups)
    ret = P.conv2d(input, weight, stride, padding, dilation, groups)
    if bias is not None:
        ret = ret + reshape(bias, (1, bias.shape[0], 1, 1))
    return ret


@core
def conv_transpose2d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Map of Pytorch method torch.nn.functional.conv_transpose2d."""
    ret = P.conv_transpose2d(
        input, weight, stride, padding, output_padding, groups, dilation
    )
    if bias is not None:
        ret = ret + reshape(bias, (1, bias.shape[0], 1, 1))
    return ret


@core
def cross_entropy(input, target, reduction="mean"):
    """Map of method torch.nn.functional.cross_entropy."""
    a = log_softmax(input, 1)
    b = nll_loss(a, target, reduction=reduction)
    return b


@core
def embedding(input, weight):
    """Map of method torch.nn.functional.embedding."""
    return P.take(weight, input)


@core
def gather(self, dim, index):
    """Map of 'gather' pytorch method."""
    return P.gather(self, dim, index)


@core
def linear(input, weight, bias=None):
    r"""Applies a linear transformation to the incoming data.

    :math:`y = xA^T + b`

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`

    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = bias + input @ weight.t()
    else:
        output = input @ weight.t()
        if bias is not None:
            output = output + bias
        ret = output
    return ret


@core
def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden
    gates = (
        P.dot(input, operations.t(w_ih))
        + P.dot(hx, operations.t(w_hh))
        + b_ih
        + b_hh
    )

    ingate, forgetgate, cellgate, outgate = chunk(gates, 4, 1)

    ingate = sigmoid(ingate)
    forgetgate = sigmoid(forgetgate)
    cellgate = operations.array_tanh(cellgate)
    outgate = sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * operations.array_tanh(cy)

    return hy, cy


@core
def log_softmax(self, dim=None, dtype=None):
    """Map of 'log_softmax' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    maxes = _max(x, dim, keepdim=True)[0]
    lse_stable = operations.array_log(
        _sum(operations.array_exp(x - maxes), dim, keepdim=True)
    )
    return x - maxes - lse_stable


@core
def item(x):
    """Map of 'item' pytorch method."""
    return P.array_to_scalar(reshape(x, ()))


# TODO 2_array_compare_max also; will probably need multitype graph
@core
def _max(self, dim=None, keepdim=False):
    """Map of 'max' pytorch method."""
    x = self
    dim_orig = dim
    if dim is None:
        dim = _build_fwd_tuple(x.shape)
    elif isinstance(dim, int):
        dim = (dim,)

    dim = _dim_tuple_explicit(x.shape, dim)

    ret_max = P.array_max(x, dim)
    ret_argmax = P.argmax(x, dim)
    if dim_orig is None:
        final_shape = ()
        return P.reshape(ret_max, final_shape)
    else:
        final_shape = _shp_squeeze(
            ret_max.shape, _dim_explicit(x.shape, dim_orig), keepdim
        )
        return (
            P.reshape(ret_max, final_shape),
            P.reshape(ret_argmax, final_shape),
        )


@core
def max_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    r"""Applies a max_pool2d."""
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    assert return_indices is False

    ret = P.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret


# TODO: mean with tuple dim (reduce over multiple chosen dims)
@core
def mean(self, dim=None, keepdim=False, *, dtype=None):
    """Map of 'mean' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, AA_bool):
        x = P.array_cast(x, i64)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    if dim is None:
        return P.array_reduce(P.scalar_add, x, ()) / P.scalar_cast(
            _total_elements(x.shape), x.dtype
        )
    else:
        return operations.array_reduce_dim(
            P.scalar_add, x, dim, keepdim
        ) / P.scalar_cast(x.shape[dim], x.dtype)


@core
def mse_loss(input, target, reduction="mean"):
    """Map of 'mse_loss' pytorch method."""
    out = (input - target) ** 2

    if reduction == "none":
        out = out
    elif reduction == "mean":
        out = mean(out)
    elif reduction == "sum":
        out = _sum(out)
    return out


# TODO:
# F.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
# reduce=None, reduction='mean')
# Is current implementation numerically stable?
@core
def nll_loss(logs, targets, reduction="mean"):
    """Map of 'nll_loss' pytorch method."""
    out = -reshape(
        gather(
            logs, 1, P.array_cast(reshape(targets, (logs.shape[0], 1)), i64)
        ),
        (logs.shape[0],),
    )

    if reduction == "none":
        out = out
    elif reduction == "mean":
        out = mean(out)
    elif reduction == "sum":
        out = _sum(out)
    return out


# TODO
# Parameter p does not currently support string values ('fro' and 'nuc')
# (issues: p expected either number and not string, or string and not number)
# Frobenius norm (p = 'fro') can be computed with p = 2.
# Nuclear norm (p = 'nuc') is not yet supported.
@core
def norm(inp, p=None, dim=None):
    """Map of torch.norm method."""
    if p is None:
        p = 2

    # Convert p to input type.
    p = P.scalar_to_array(P.scalar_cast(p, inp.dtype), operations.typeof(inp))
    # Create a value 1 with input type.
    one = P.scalar_to_array(P.scalar_cast(1, inp.dtype), operations.typeof(inp))

    # Reshape inp based on dims, making sure that dimensions to norm
    # are all assembled in latest inp dimension.
    leading_dims = None
    leading_shape = None

    if dim is None:
        # We must norm the entire tensor.
        new_inp = reshape(inp, (1, -1))
    else:
        # Check input shape and given dimensions to retrieve
        # leading dimensions (not to norm),
        # permutation to apply to input if necessary,
        # leading shape,
        # and reshaping tuple (leading shape + (-1,)), to apply to
        # permuted input if necessary.
        (
            leading_dims,
            permutation,
            leading_shape,
            reshaping_tuple,
        ) = _prepare_dims_to_norm(inp.shape, dim)

        # Now we can reshape inp.
        if leading_dims is None:
            # We must norm the entire tensor.
            new_inp = reshape(inp, (1, -1))
        else:
            # We must norm only dimensions in normed_dims.
            # 1) Move dimensions to norm to the end of tensor dimension.
            # 2) Reshape to pack trailing dimensions to norm into 1 dimension.
            new_inp = P.transpose(inp, permutation)
            new_inp = reshape(new_inp, *reshaping_tuple)

    # Then we can compute norm.
    if p.item() == np.inf:
        # Maximum of absolute values
        res = _max(operations.array_abs(new_inp), -1)[0]
    elif p.item() == -np.inf:
        # Minimum of absolute values.
        # res = new_inp.abs().min(-1)[0]
        u = operations.array_abs(new_inp)
        v = -u
        w = _max(v, -1)[0]
        res = -w
    else:
        # Classical p-norm.
        # res = (new_inp.abs() ** p_value).sum(-1) ** (1 / p_value)
        a = operations.array_abs(new_inp) ** p
        b = _sum(a, -1)
        c = one / p
        res = b ** c
    if leading_dims is not None:
        res = reshape(res, leading_shape)
    return res


@core
def relu(x):
    """Relu activation function."""
    return hyper_map(P.scalar_max, x, 0.0)


@core
def reshape(x, *shp):
    """Reshape that allow unknown dim (-1)."""
    if len(shp) == 1:
        if isinstance(shp[0], tuple):
            shp = shp[0]
    return P.reshape(x, _shp_explicit(x.shape, shp))


@core
def scatter(self, dim, index, src):
    """Map of 'scatter' pytorch method."""
    if P.hastype(src, AS):
        src = P.scalar_to_array(
            P.scalar_cast(src, self.dtype), operations.typeof(self)
        )
    if len(src.shape) == 0:
        src = P.distribute(src, index.shape)
    return P.scatter(self, dim, index, src)


@core
def scatter_add(self, dim, index, src):
    """Map of 'scatter_add' pytorch method."""
    return P.scatter_add(self, dim, index, src)


@core
def sigmoid(x):
    """Sigmoid activation function."""
    return (operations.array_tanh(x / 2) + 1) / 2


@core
def size(self, dim=None):
    """Map of 'size' pytorch method."""
    if dim is None:
        return self.shape
    else:
        return self.shape[dim]


@core
def smooth_l1_loss(input, target, reduction="mean"):
    """Map of 'smooth_l1_loss' pytorch method."""
    z_raw = input - target
    z_abs = operations.array_abs(z_raw)

    def pw(_z_abs):
        out = P.switch(_z_abs < 1, 0.5 * (_z_abs ** 2), _z_abs - 0.5)
        return out

    out = P.array_map(pw, z_abs)

    if reduction == "none":
        out = out
    elif reduction == "mean":
        out = mean(out)
    elif reduction == "sum":
        out = _sum(out)
    return out


@core
def softmax(self, dim=None, dtype=None):
    """Map of 'softmax' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    maxes = _max(x, dim, keepdim=True)[0]
    x_exp = operations.array_exp(x - maxes)
    x_exp_sum = _sum(x_exp, dim, keepdim=True)
    return x_exp / x_exp_sum


# TODO: support for split_size rather than just sections
@core
def split(self, split_size_or_sections, dim=0):
    """Map of 'split' pytorch method."""
    return P.split(self, split_size_or_sections, dim)


@core
def squeeze(self, dim=None):
    """Map of 'squeeze' pytorch method."""
    dim = _dim_explicit(self.shape, dim)
    final_shape = _shp_squeeze(self.shape, dim, False)
    return reshape(self, final_shape)


@core
def stack(self, dim=0):
    """Map of 'stack' pytorch method."""
    x = self
    x_u = ()
    for _x in x:
        x_u = x_u + (unsqueeze(_x, dim),)
    return P.concat(x_u, dim)


# TODO: std with tuple dim (reduce over multiple chosen dims)
@core
def std(self, dim=None, unbiased=True, keepdim=False, *, dtype=None):
    """Map of 'std' pytorch method."""
    return (
        var(self, dim=dim, unbiased=unbiased, keepdim=keepdim, dtype=dtype)
        ** 0.5
    )


@core
def _sum(self, dim=None, keepdim=False, *, dtype=None):
    """Map of 'sum' pytorch method."""
    x = self
    if isinstance(dim, tuple):
        dim = _dim_tuple_explicit(x.shape, dim)
    else:
        dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, AA_bool):
        x = P.array_cast(x, i64)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    if dim is None:
        return P.array_reduce(P.scalar_add, x, ())
    else:
        return operations.array_reduce_dim(P.scalar_add, x, dim, keepdim)


@core
def tensor_dim(t):
    """Map of 'dim' pytorch method."""
    return len(P.shape(t))


@myia_static
def _transpose_dims(a_dims, dim0, dim1):
    dims = ()
    for d in range(a_dims):
        if d == dim0:
            dims = dims + (dim1,)
        elif d == dim1:
            dims = dims + (dim0,)
        else:
            dims = dims + (d,)
    return dims


@core
def transpose(a, dim0, dim1):
    """Map of 'transpose' pytorch method."""
    dims = _transpose_dims(len(a.shape), dim0, dim1)
    return P.transpose(a, dims)


@core
def uniform(rstate, size, _min, _max, dtype=f32):
    """Returns samples from uniform distribution bounded by _min and _max"""
    r0, v0 = P.random_uint32(rstate, size)
    _min = P.scalar_to_array(_min, AA)
    _max = P.scalar_to_array(_max, AA)
    _min = P.array_cast(_min, dtype)
    _max = P.array_cast(_max, dtype)
    rand_range = _max - _min
    v0 = P.array_cast(v0, dtype)
    return (v0 * (rand_range / 4294967296)) + _min, r0


@core
def unsqueeze(self, dim=None):
    """Map of 'unsqueeze' pytorch method."""
    dim = _dim_explicit_unsqueeze(self.shape, dim)
    final_shape = _shp_unsqueeze(self.shape, dim)
    return reshape(self, final_shape)


# TODO: var with tuple dim (reduce over multiple chosen dims)
@core
def var(self, dim=None, unbiased=True, keepdim=False, *, dtype=None):
    """Map of 'var' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, AA_bool):
        x = P.array_cast(x, i64)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    x = (x - mean(x, dim=dim, keepdim=True, dtype=None)) ** 2
    x = _sum(x, dim=dim, keepdim=keepdim, dtype=None)

    denom = _var_denom(self.shape, dim, unbiased)

    denom = P.scalar_cast(denom, x.dtype)

    return x / denom


@core
def view_as(x, y):
    """Map of 'view_as' pytorch method."""
    return P.reshape(x, y.shape)


__all__ = ["conv2d", "cross_entropy", "item", "linear", "relu", "sigmoid"]
