"""PyTorch Frontend."""

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

import torch

from .. import operations
from ..abstract import build_value, macro, myia_static
from ..hypermap import hyper_map
from ..ir import Constant
from ..operations import primitives as P
from ..utils import MyiaValueError, core
from ..xtype import TupleT, f32, i64, u64
from .pytorch_abstract_types import APT, APT_bool

# ############# THESE FUNCTIONS SHOULD BE IN ALPHABETICAL ORDER #############


# HELPER FUNCTIONS ##########################################################


@macro
async def _denom(info, shp_ref, dtype_ref, dim_ref, unbiased_ref):
    shp = build_value(await shp_ref.get())
    dim = build_value(await dim_ref.get())
    unbiased = build_value(await unbiased_ref.get())

    def prod(_x):
        return reduce(operator.mul, _x, 1)

    if dim is None:
        denom = int(prod(shp))
    else:
        denom = shp[dim]

    if unbiased is True:
        denom = denom - 1

    return Constant(denom)


@macro
async def _dim_explicit(info, a_shp_ref, dim_ref):
    a_shp = build_value(await a_shp_ref.get())
    dim = build_value(await dim_ref.get())
    if dim == -1:
        dim = len(a_shp) - 1
    return Constant(dim)


@macro
async def _dim_tuple_explicit(info, a_shp_ref, dim_ref):
    a_shp = build_value(await a_shp_ref.get())
    dim = build_value(await dim_ref.get())
    shp_explicit = ()
    for s in dim:
        if s == -1:
            shp_explicit = shp_explicit + (len(a_shp) - 1,)
        else:
            shp_explicit = shp_explicit + (s,)
    return Constant(shp_explicit)


@myia_static
def _build_fwd_tuple(shp):
    t = ()
    for d in range(len(shp)):
        t = t + (d,)
    return t


@core
def _ensure_u64(x):
    assert (P.hastype(x, i64) or P.hastype(x, u64))
    assert x >= 0
    return P.scalar_cast(x, u64)


@core
def _pair(x):
    if not P.hastype(x, TupleT):
        x = (x, x)
    x = (_ensure_u64(x[0]), x[1])
    x = (x[0], _ensure_u64(x[1]))
    return x


@macro
async def _shp_squeeze(info, o_shp_ref, dim_ref, keepdim_ref):
    orig_shp = build_value(await o_shp_ref.get())
    dim = build_value(await dim_ref.get())
    keepdim = build_value(await keepdim_ref.get())

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
    return Constant(final_shape)


@macro
async def _shp_explicit(info, a_shp_ref, shp_ref):
    a_shp = build_value(await a_shp_ref.get())
    shp = build_value(await shp_ref.get())

    def prod(_x):
        return reduce(operator.mul, _x, 1)

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

    if (prod(a_shp) % prod(shp) != 0 if unk_count == 1
            else prod(shp) != prod(a_shp)):
        e_msg = "Cannot change the total number of elements in reshape"
        raise MyiaValueError(e_msg)

    known_unk_dim = int(abs(prod(a_shp) / prod(shp)))
    shp_explicit = ()
    for s in shp:
        if s == -1:
            shp_explicit = shp_explicit + (known_unk_dim, )
        else:
            shp_explicit = shp_explicit + (s, )

    return Constant(shp_explicit)


@macro
async def _total_elements(info, shp_ref):
    shp = build_value(await shp_ref.get())

    def prod(_x):
        return reduce(operator.mul, _x, 1)

    _tot_elems = int(prod(shp))

    return Constant(_tot_elems)


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
    final_shape = _shp_squeeze(ret.shape,
                               _dim_explicit(x.shape, dim_orig),
                               keepdim)
    return P.reshape(ret, final_shape)


@core
def cat(self, dim=0):
    """Map of 'cat' pytorch method."""
    x = self
    return P.concat(x, dim)


@core
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
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
        ret = ret + bias.reshape((1, bias.shape[0], 1, 1))
    return ret


@core
def gather(self, dim, index):
    """Map of 'gather' pytorch method."""
    return P.gather(self, dim, index)


@core
def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

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
def log_softmax(self, dim=None, dtype=None):
    """Map of 'log_softmax' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    maxes = torch.max(x, dim, keepdim=True)[0]
    lse_stable = torch.log(torch.sum(torch.exp(x - maxes), dim, keepdim=True))
    return x - maxes - lse_stable


@core
def item(x):
    """Map of 'item' pytorch method."""
    return P.array_to_scalar(x.reshape(()))


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
        final_shape = _shp_squeeze(ret_max.shape,
                                   _dim_explicit(x.shape, dim_orig),
                                   keepdim)
        return (P.reshape(ret_max, final_shape),
                P.reshape(ret_argmax, final_shape))


@core
def max_pool2d(input, kernel_size, stride=(), padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a max_pool2d."""
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    ret = P.max_pool2d(input, kernel_size, stride, padding, dilation,
                       ceil_mode)
    if not return_indices:
        ret = ret[0]
    return ret


# TODO: mean with tuple dim (reduce over multiple chosen dims)
@core
def mean(self, dim=None, keepdim=False, *, dtype=None):
    """Map of 'mean' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, APT_bool):
        x = P.array_cast(x, i64)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    if dim is None:
        return P.array_reduce(P.scalar_add, x, ()) \
            / P.scalar_cast(_total_elements(x.shape), x.dtype)
    else:
        return operations.array_reduce_dim(P.scalar_add, x, dim, keepdim) \
            / P.scalar_cast(x.shape[dim], x.dtype)


# TODO:
# F.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
# reduce=None, reduction='mean')
# Is current implementation numerically stable?
@core
def nll_loss(logs, targets, reduction='mean'):
    """Map of 'nll_loss' pytorch method."""
    out = -torch.gather(
        logs,
        1,
        targets.reshape((logs.shape[0], 1))).reshape((logs.shape[0],))

    if reduction == 'none':
        out = out
    elif reduction == 'mean':
        out = torch.sum(out) / P.scalar_cast(logs.shape[0], out.dtype)
    elif reduction == 'sum':
        out = torch.sum(out)
    return out


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
    if not P.hastype(src, APT):
        src = P.scalar_to_array(P.scalar_cast(src, self.dtype), APT)
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
    x = self
    dim = _dim_explicit(x.shape, dim)
    if dim is None:
        return x.shape
    else:
        return x.shape[dim]


@macro
async def get_z(info, z_ref):
    z_abs = build_value(await z_ref.get())

    def pw(_z_abs):
        if _z_abs < 1:
            #out = 0.5 * (_z_raw ** 2)
            out = 0.5 * (_z_abs ** 2)
        else:
            out = _z_abs - 0.5
        return out

    out = P.array_map(pw, z_abs)

    return Constant(out)


@core
def smooth_l1_loss(input, target, reduction='mean'):
    """Map of 'smooth_l1_loss' pytorch method."""

    z_raw = input - target
    z_abs = torch.abs(z_raw)

    """
    if z_abs < 1:
        out = 0.5 * (z_raw ** 2)
    else:
        out = z_abs - 0.5
        #"""

    # def pw(_z_raw, _z_abs):

    """
    def pw(_z_abs):
        if _z_abs < 1:
            #out = 0.5 * (_z_raw ** 2)
            out = 0.5 * (_z_abs ** 2)
        else:
            out = _z_abs - 0.5
        return out

    out = P.array_map(pw, z_abs)
    #"""

    out = get_z(z_abs)

    if reduction == 'none':
        out = out
    elif reduction == 'mean':
        out = torch.sum(out) / P.scalar_cast(input.shape[0], out.dtype)
    elif reduction == 'sum':
        out = torch.sum(out)
    # return out
    raise NotImplementedError()


@core
def softmax(self, dim=None, dtype=None):
    """Map of 'softmax' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp / x_exp_sum


#TODO: support for split_size rather than just sections
@core
def split(self, split_size_or_sections, dim=0):
    """Map of 'split' pytorch method."""
    x = self
    return P.split(x, split_size_or_sections, dim)


@core
def squeeze(self, dim=None):
    """Map of 'squeeze' pytorch method."""
    final_shape = _shp_squeeze(self.shape, dim, False)
    return self.reshape(final_shape)


# TODO: std with tuple dim (reduce over multiple chosen dims)
@core
def std(self, dim=None, unbiased=True, keepdim=False, *, dtype=None):
    """Map of 'std' pytorch method."""
    return torch.var(self,
                     dim=dim, unbiased=unbiased,
                     keepdim=keepdim, dtype=dtype) ** .5


# TODO: sum with tuple dim (reduce over multiple chosen dims)
@core
def _sum(self, dim=None, keepdim=False, *, dtype=None):
    """Map of 'sum' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, APT_bool):
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


# TODO: var with tuple dim (reduce over multiple chosen dims)
@core
def var(self, dim=None, unbiased=True, keepdim=False, *, dtype=None):
    """Map of 'var' pytorch method."""
    x = self
    dim = _dim_explicit(x.shape, dim)

    if P.hastype(x, APT_bool):
        x = P.array_cast(x, i64)

    if dtype is not None:
        x = P.array_cast(x, dtype)

    x = (x - x.mean(dim=dim, keepdim=True, dtype=None)) ** 2
    x = torch.sum(x, dim=dim, keepdim=keepdim, dtype=None)

    denom = _denom(self.shape, x.dtype, dim, unbiased)

    denom = P.scalar_cast(denom, x.dtype)

    return x / denom


@core
def view_as(x, y):
    """Map of 'view_as' pytorch method."""
    return P.reshape(x, y.shape)


@core
def zeros(*shp, dtype=None):
    """Map of 'dim' pytorch method."""
    if dtype is None:
        dtype = f32

    if len(shp) == 1:
        if isinstance(shp[0], tuple):
            shp = shp[0]
    return P.distribute(P.scalar_to_array(P.scalar_cast(0.0, dtype), APT), shp)


__all__ = [
    'conv2d',
    'item',
    'linear',
    'relu',
    'sigmoid',
    'zeros',
]
