"""Linear implementation using pytorch."""

import numpy as np
import torch

from ...prim import Primitive, ops as P
from ...xtype import Bool, Float, Int, UInt, type_to_np_dtype
from ...ir import manage
from ...prim import Primitive, ops as P
from ..transform import CompileGraphs, nonlinear_ops
from . import Backend, HandleBackend
from .pytorch_conv_grad import conv2d_input, conv2d_weight

_type_map = {
    Int[8]: torch.int8,
    Int[16]: torch.int16,
    Int[32]: torch.int32,
    Int[64]: torch.int64,
    UInt[8]: torch.uint8,
    Float[16]: torch.float16,
    Float[32]: torch.float32,
    Float[64]: torch.float64,
    Bool: torch.uint8,
    # This is a hack but we really need uint64 support
    UInt[64]: torch.int64,
}


def pytorch_array_to_scalar(v):
    """Implementation of array_to_scalar for pytorch."""
    if v.is_cuda:
        v = v.cpu()
    return v.detach().numpy()


simple_mapping = {
    P.scalar_add: lambda a, b: a + b,
    P.scalar_sub: lambda a, b: a - b,
    P.scalar_mul: lambda a, b: a * b,
    P.scalar_div: lambda a, b: (a / b).astype(a.dtype),
    P.scalar_mod: lambda a, b: a % b,
    P.scalar_pow: lambda a, b: a ** b,
    P.scalar_floor: np.floor,
    P.scalar_uadd: lambda a: a,
    P.scalar_usub: lambda a: -a,
    P.scalar_exp: np.exp,
    P.scalar_log: np.log,
    P.scalar_tan: np.tan,
    P.scalar_tanh: np.tanh,

    P.scalar_eq: lambda a, b: a == b,
    P.scalar_lt: lambda a, b: a < b,
    P.scalar_gt: lambda a, b: a > b,
    P.scalar_ne: lambda a, b: a != b,
    P.scalar_le: lambda a, b: a <= b,
    P.scalar_ge: lambda a, b: a >= b,

    P.bool_and: lambda a, b: a & b,
    P.bool_or: lambda a, b: a | b,
    P.bool_eq: lambda a, b: a == b,
    P.bool_not: lambda a: ~a,

    P.distribute: lambda a, shp: a.expand(*shp),
    P.transpose: lambda a, perm: a.permute(*perm),
    P.reshape: lambda a, shp: a.reshape(shp),
    P.dot: torch.mm,

    P.array_to_scalar: pytorch_array_to_scalar,
}


scalar_mapping = {
    P.scalar_add: lambda a, b: a + b,
    P.scalar_sub: lambda a, b: a - b,
    P.scalar_mul: lambda a, b: a * b,
    P.scalar_div: lambda a, b: a / b,
    P.scalar_mod: lambda a, b: a % b,
    P.scalar_pow: lambda a, b: a ** b,
    P.scalar_floor: torch.floor,
    P.scalar_uadd: lambda a: a,
    P.scalar_usub: lambda a: -a,
    P.scalar_exp: torch.exp,
    P.scalar_log: torch.log,
    P.scalar_tan: torch.tan,
    P.scalar_tanh: torch.tanh,

    P.scalar_eq: torch.eq,
    P.scalar_lt: torch.lt,
    P.scalar_gt: torch.gt,
    P.scalar_ne: torch.ne,
    P.scalar_le: torch.le,
    P.scalar_ge: torch.ge,

    P.bool_and: lambda a, b: a & b,
    P.bool_or: lambda a, b: a | b,
    P.bool_eq: torch.eq,
    P.bool_not: lambda a: ~a,
}


def pytorch_scalar_cast(op):
    v = op.inputs[1]
    assert op.inputs[2].is_constant()
    dtype = type_to_np_dtype(op.inputs[2].value)

    def _impl(v):
        return (v.astype(dtype),)
    return _impl, (v,)


def pytorch_array_map(op):
    """Implementation of array_map for pytorch."""
    fn = op.inputs[1]
    assert fn.is_constant(Primitive)
    fn = fn.value
    if fn in scalar_mapping:
        impl = scalar_mapping[fn]
    else:
        raise NotImplementedError(f'array_map of {fn}')

    def _impl(*args):
        return (impl(*args),)
    return _impl, op.inputs[2:]


def pytorch_array_reduce(op):
    """Implementation of array_reduce for pytorch."""
    fn = op.inputs[1]
    shape = op.inputs[3]
    assert fn.is_constant(Primitive)
    assert shape.is_constant(tuple)
    fn = fn.value
    tshp = shape.value

    if fn == P.scalar_add:
        impl = torch.sum
    else:
        raise NotImplementedError(f"reduce with {fn}")

    def _impl(array):
        ashp = array.shape

        if len(tshp) < len(ashp):
            ts = (1,) * (len(ashp) - len(tshp)) + tshp
        else:
            ts = tshp
        axis = list(i for i, t in enumerate(ts) if t == 1)
        if len(axis) == 1:
            axis = axis[0]
        res = impl(array, axis, keepdim=True)
        if len(tshp) < len(ashp):
            res = torch.reshape(res, shape=tshp)
        return (res,)
    return _impl, (op.inputs[2],)

#############################################################################


def conv2d_wrap(input, weight, stride, padding, dilation, groups):
    """Wrap of conv2d for pytorch."""
    groups = groups.item()
    return (torch.nn.functional.conv2d(
        input, weight, None, stride, padding, dilation, groups),)


def pytorch_conv2d(op):
    """Implementation of conv2d for pytorch."""
    return conv2d_wrap, op.inputs[1:]

#############################################################################


def conv2d_input_grad_wrap(input_size, weight, grad_output, stride, padding,
                           dilation, groups):
    """Wrap of conv2d_input_grad for pytorch."""
    input_size = tuple(i.item() for i in input_size)
    stride = tuple(_x.item() for _x in stride)
    padding = tuple(_x.item() for _x in padding)
    dilation = tuple(_x.item() for _x in dilation)
    groups = groups.item()
    return (conv2d_input(
        input_size, weight, grad_output, stride, padding, dilation, groups),)


def pytorch_conv2d_input_grad(op):
    """Implementation of conv2d_input_grad for pytorch."""
    return conv2d_input_grad_wrap, op.inputs[1:]

#############################################################################


def conv2d_weight_grad_wrap(input, weight_size, grad_output, stride, padding,
                            dilation, groups):
    """Wrap of conv2d_weight_grad for pytorch."""
    weight_size = tuple(w.item() for w in weight_size)
    stride = tuple(_x.item() for _x in stride)
    padding = tuple(_x.item() for _x in padding)
    dilation = tuple(_x.item() for _x in dilation)
    groups = groups.item()
    return (conv2d_weight(
        input, weight_size, grad_output, stride, padding, dilation, groups),)


def pytorch_conv2d_weight_grad(op):
    """Implementation of conv2d_weight_grad for pytorch."""
    return conv2d_weight_grad_wrap, op.inputs[1:]

#############################################################################


_mapping = {
    P.array_map: pytorch_array_map,
    P.array_reduce: pytorch_array_reduce,
    P.conv2d: pytorch_conv2d,
    P.conv2d_input_grad: pytorch_conv2d_input_grad,
    P.conv2d_weight_grad: pytorch_conv2d_weight_grad,
    P.scalar_cast: pytorch_scalar_cast,
}

for k, v in simple_mapping.items():
    _mapping[k] = lambda op, v=v: (lambda *args: (v(*args),), op.inputs[1:])


def pytorch_convert(lst, backend):
    """Convert myia op to pytorch op."""
    assert len(lst) == 1
    op = lst[0]

    assert op.is_apply()
    assert op.inputs[0].is_constant(Primitive)

    fn = op.inputs[0].value
    if fn == P.scalar_to_array:
        # Hack because we need the runtime context here.
        return lambda v: (backend.from_numpy(v),), [op.inputs[1]], [op]

    mapper = _mapping.get(fn, None)
    if mapper is None:
        raise NotImplementedError(fn)
    impl, inputs = mapper(op)
    return impl, inputs, [op]


class PyTorchBackend(Backend):
    """Backend to run using pytorch.

    Backend options:
        device: the target device for data storage ('cpu', 'cuda', 'cuda:X')

    """

    def __init__(self, device):
        """Create a PyTorch backend on the given device."""
        if device == 'cuda':
            device = 'cuda:0'
        self.device = torch.device(device)
        self.compiler = CompileGraphs(lambda lst: pytorch_convert(lst, self),
                                      nonlinear_ops, self, split_linear=True)

    def compile(self, graph, *others):
        """Compile a graph."""
        manage(graph)
        return self.compiler.compile_and_link(graph)

    def to_numpy(self, v):
        """Make a numpy array from a torch tensor."""
        if v.is_cuda:
            v = v.cpu()
        return v.detach().numpy()

    def from_numpy(self, a):
        """Make a torch tensor from a numpy array."""
        return torch.from_numpy(a).to(self.device)

    def to_scalar(self, v):
        """Convert a torch tensor to a scalar."""
        if (v is None) or (v is True) or (v is False) or (isinstance(v, str)):
            return v
        else:
            return v.item()

    def from_scalar(self, s, t):
        """Convert a scalar to a torch tensor."""
        if s is None:
            return None
        dt = type_to_np_dtype(t)
        return np.asarray(s, dtype=dt)


class PyTorchBackendR(HandleBackend):
    """Pytorch proxy."""

    def __init__(self, device='cpu'):
        self.real = PyTorchBackend(device)
