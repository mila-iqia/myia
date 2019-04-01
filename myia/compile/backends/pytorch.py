"""Linear implementation using pytorch."""

import torch

from . import Backend
from ..transform import CompileGraphs, nonlinear_ops

from ...dtype import Int, UInt, Float
from ...prim import Primitive, ops as P


_type_map = {
    Int[8]: torch.int8,
    Int[16]: torch.int16,
    Int[32]: torch.int32,
    Int[64]: torch.int64,
    UInt[8]: torch.uint8,
    Float[16]: torch.float16,
    Float[32]: torch.float32,
    Float[64]: torch.float64,
}


def type_to_pytorch_type(t):
    """Map myia types to pytorch types."""
    if t not in _type_map:
        raise TypeError(f"Unsupported type: {t}")
    return _type_map[t]


def pytorch_array_map(fn, *arrays):
    """Implementation of array_map for pytorch."""
    assert fn.is_constant(Primitive)
    fn = fn.value
    if fn in _mapping:
        return _mapping[fn](*arrays)


def pytorch_array_reduce(fn, array, shape):
    """Implementation of array_reduce for pytorch."""
    assert fn.is_constant(Primitive)
    assert shape.is_constant(tuple)
    fn = fn.value
    tshp = shape.value
    ashp = array.shape
    if len(tshp) < len(ashp):
        ts = (1,) * (len(ashp) - len(tshp)) + tshp
    else:
        ts = tshp
    axis = list(i for i, t in enumerate(ts) if t == 1)
    if len(axis) == 1:
        axis = axis[0]
    if fn == P.scalar_add:
        res = torch.sum(array, axis)
    else:
        raise NotImplementedError(f"reduce with {fn}")
    if len(tshp) < len(ashp):
        res = torch.reshape(res, shape=tshp)
    return res


_mapping = {
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

    P.scalar_eq: lambda a, b: a == b,
    P.scalar_lt: lambda a, b: a < b,
    P.scalar_gt: lambda a, b: a > b,
    P.scalar_ne: lambda a, b: a != b,
    P.scalar_le: lambda a, b: a <= b,
    P.scalar_ge: lambda a, b: a >= b,

    P.bool_and: lambda a, b: a and b,
    P.bool_or: lambda a, b: a or b,
    P.bool_eq: lambda a, b: a == b,
    P.bool_not: lambda a: not a,

    P.distribute: lambda a, shp: a.expand(*shp),
    P.transpose: lambda a, perm: a.permute(*perm),
    P.dot: torch.mm,

    P.scalar_to_array: lambda x: x,
    P.array_map: pytorch_array_map,
    P.array_reduce: pytorch_array_reduce,
}


def pytorch_convert(lst):
    """Convert myia op to pytorch op."""
    assert len(lst) == 1
    op = lst[0]

    assert op.is_apply()
    assert op.inputs[0].is_constant(Primitive)

    fn = op.inputs[0].value
    res = _mapping.get(fn, None)
    if res is None:
        raise NotImplementedError(fn)
    return res


class PyTorchBackend(Backend):
    """Backend to run using pytorch.

    Backend options:
        device: the target device for data storage ('cpu', 'cuda', 'cuda:X')

    """

    def __init__(self, device='cpu'):
        """Create a PyTorch backend on the given device."""
        self.device = torch.device(device)
        self.compiler = CompileGraphs(pytorch_convert, nonlinear_ops, self,
                                      split_linear=True)

    def compile(self, graph):
        """Compile a graph."""
        return self.compiler.compile_and_link(graph)

    def to_numpy(self, v):
        """Make a numpy array from a torch tensor."""
        return v.numpy()

    def from_numpy(self, a):
        """Make a torch tensor from a numpy array."""
        return torch.from_numpy(a).to(self.device)

    def to_scalar(self, v):
        """Convert a torch tensor to a scalar."""
        return v.item()

    def from_scalar(self, s, t):
        """Convert a scalar to a torch tensor."""
        dt = type_to_pytorch_type(t)
        return torch.tensor(s, dtype=dt, device=self.device)

    def to_dlpack(self, v):
        """Make a dlpack capsule from a torch tensor."""
        return torch.utils.dlpack.to_dlpack(v)

    def from_dlpack(self, dlp):
        """Make a torch tensor from a dlpack capsule."""
        return torch.utils.dlpack.from_dlpack(dlp).to(self.device)

    def check_array(self, v, t):
        if not isinstance(v, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
        if v.device != self.device:  # pragma: no cover
            raise RuntimeError("Tensor on wrong device.")
        if v.dtype != type_to_pytorch_type(t):
            raise TypeError("Wrong dtype")
