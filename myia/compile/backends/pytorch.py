"""Linear implementation using pytorch."""

import numpy as np
import torch

from ... import abstract, xtype
from ...ir import manage
from ...operations import Primitive, primitives as P
from ...utils import RandomStateWrapper, TaggedValue, untested_legacy
from ...xtype import Bool, Float, Int, UInt, type_to_np_dtype
from ..cconv import closure_convert
from ..transform import CompileGraphs, nonlinear_ops
from . import Backend, HandleBackend
from .pytorch_conv_grad import conv2d_weight

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


def pytorch_take_grad_inp(nb_indices, indices, values):
    row_size = values.shape[-1]
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    output = torch.zeros((nb_indices, row_size), dtype=values.dtype)
    for i in range(nb_indices):
        output[i] = (
            ((broadcastable_indices == i).to(values.dtype) * values)
            .reshape((-1, row_size))
            .sum(dim=(0,))
        )
    return output


def pytorch_random_initialize(seed):
    """Implementation of random_initialize for pytorch."""
    rng = torch.Generator()
    rng.manual_seed(seed.item())
    return rng.get_state()


def pytorch_random_uint32(rstate, shape):
    """Implementation of random_uint32 for pytorch."""
    shape = tuple(dim.item() for dim in shape)
    rng = torch.Generator()
    rng.set_state(rstate)
    output = torch.zeros(shape, dtype=torch.int64)
    output.random_(0, 2 ** 32, generator=rng)
    return rng.get_state(), output


simple_mapping = {
    P.scalar_abs: np.absolute,
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
    P.scalar_sin: np.sin,
    P.scalar_cos: np.cos,
    P.scalar_trunc: np.trunc,
    P.scalar_eq: lambda a, b: a == b,
    P.scalar_lt: lambda a, b: a < b,
    P.scalar_gt: lambda a, b: a > b,
    P.scalar_ne: lambda a, b: a != b,
    P.scalar_le: lambda a, b: a <= b,
    P.scalar_ge: lambda a, b: a >= b,
    P.scalar_sign: np.sign,
    P.scalar_bit_and: lambda a, b: a & b,
    P.scalar_bit_or: lambda a, b: a | b,
    P.scalar_bit_xor: lambda a, b: a ^ b,
    P.scalar_bit_lshift: lambda a, b: a << b,
    P.scalar_bit_rshift: lambda a, b: a >> b,
    P.scalar_bit_not: lambda a: ~a,
    P.bool_and: lambda a, b: a & b,
    P.bool_or: lambda a, b: a | b,
    P.bool_eq: lambda a, b: a == b,
    P.bool_not: lambda a: ~a,
    P.distribute: lambda a, shp: a.expand(*shp) if shp != () else a,
    P.transpose: lambda a, perm: a.permute(*perm),
    P.reshape: lambda a, shp: a.reshape(shp),
    P.dot: torch.mm,
    P.take: lambda w, i: torch.nn.functional.embedding(i, w),
    P.take_grad_inp: pytorch_take_grad_inp,
    P.array_to_scalar: pytorch_array_to_scalar,
    P.random_initialize: pytorch_random_initialize,
    P.random_uint32: pytorch_random_uint32,
}


scalar_mapping = {
    P.scalar_abs: torch.abs,
    P.scalar_add: lambda a, b: a + b,
    P.scalar_sub: lambda a, b: a - b,
    P.scalar_max: torch.max,
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
    P.scalar_sin: torch.sin,
    P.scalar_cos: torch.cos,
    P.scalar_trunc: torch.trunc,
    P.scalar_eq: torch.eq,
    P.scalar_lt: torch.lt,
    P.scalar_gt: torch.gt,
    P.scalar_ne: torch.ne,
    P.scalar_le: torch.le,
    P.scalar_ge: torch.ge,
    P.scalar_sign: torch.sign,
    P.scalar_bit_and: lambda a, b: a & b,
    P.scalar_bit_or: lambda a, b: a | b,
    P.scalar_bit_xor: lambda a, b: a ^ b,
    P.scalar_bit_lshift: lambda a, b: a << b,
    P.scalar_bit_rshift: lambda a, b: a >> b,
    P.scalar_bit_not: lambda a: ~a,
    P.bool_and: lambda a, b: a & b,
    P.bool_or: lambda a, b: a | b,
    P.bool_eq: torch.eq,
    P.bool_not: lambda a: ~a,
    P.switch: torch.where,
}


def pytorch_scalar_cast(op):
    """Implementation of scalar_cast."""
    v = op.inputs[1]
    assert op.inputs[2].is_constant()
    dtype = type_to_np_dtype(op.inputs[2].value.xtype())

    def _impl(v):
        return (v.astype(dtype),)

    return _impl, (v,)


def pytorch_array_cast(op):
    """Implementation of array_cast for pytorch."""
    t = op.inputs[2]
    dt = _type_map[t.value.xtype()]

    def _impl(x):
        return (x.to(dtype=dt),)

    return _impl, op.inputs[1:2]


def pytorch_array_map(op):
    """Implementation of array_map for pytorch."""
    fn = op.inputs[1]
    assert fn.is_constant(Primitive)
    fn = fn.value
    if fn in scalar_mapping:
        impl = scalar_mapping[fn]
    else:
        raise NotImplementedError(f"array_map of {fn}")

    def _impl(*args):
        return (impl(*args),)

    return _impl, op.inputs[2:]


def _pytorch_array_reduce_add(tshp):
    """Generate implementation for sum reduction based on given axes."""

    def _impl(array):
        ashp = array.shape

        if len(tshp) < len(ashp):
            ts = (1,) * (len(ashp) - len(tshp)) + tshp
        else:
            ts = tshp
        axis = list(i for i, t in enumerate(ts) if t == 1)
        if len(axis) == 1:
            axis = axis[0]
        res = torch.sum(array, axis, keepdim=True)
        if len(tshp) < len(ashp):
            res = torch.reshape(res, shape=tshp)
        return (res,)

    return _impl


def _pytorch_array_reduce_mul(tshp):
    """Generate implementation for product reduction based on given axes."""

    def _impl(array):
        ashp = array.shape

        if len(tshp) in (0, len(ashp)):
            res = torch.prod(array)
        else:
            raise NotImplementedError(
                "We currently only support full product on an array."
            )
        return (res,)

    return _impl


def pytorch_array_reduce(op):
    """Implementation of array_reduce for pytorch."""
    fn = op.inputs[1]
    shape = op.inputs[3]
    assert fn.is_constant(Primitive)
    assert shape.is_constant(tuple)
    fn = fn.value
    tshp = shape.value

    if fn == P.scalar_add:
        gen_impl = _pytorch_array_reduce_add
    elif fn == P.scalar_mul:
        gen_impl = _pytorch_array_reduce_mul
    else:
        raise NotImplementedError(f"reduce with {fn}")

    return gen_impl(tshp), (op.inputs[2],)


def pytorch_array_getitem(op):
    """Implementation of array_getitem for pytorch."""

    def _impl(array, begin, end, strides):
        idx = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
        return (array[idx],)

    return _impl, op.inputs[1:]


def pytorch_array_setitem(op):
    """Implementation of array_setitem for pytorch."""

    def _impl(array, begin, end, strides, value):
        idx = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
        ret = array.clone()
        ret[idx] = value
        return (ret,)

    return _impl, op.inputs[1:]


def pytorch_argmax(op):
    """Implementation of argmax for pytorch."""

    def _impl(x, dim):
        dim = tuple(sorted(dim))
        n = ()
        for _s in range(len(x.shape)):
            if _s not in dim:
                n = n + (_s,)
        n = n + dim
        x = x.permute(n)
        ns = x.shape[0 : -len(dim)] + (-1,)
        r = torch.argmax(x.reshape(ns), -1, keepdim=False)
        rl = list(r.shape)
        for _sd in dim:
            rl.insert(_sd, 1)
        rf = tuple(rl)
        return (torch.reshape(r, rf),)

    return _impl, op.inputs[1:]


def pytorch_array_max(op):
    """Implementation of array_max for pytorch."""

    def _impl(x, dim):
        dim = tuple(sorted(dim))
        n = ()
        for _s in range(len(x.shape)):
            if _s not in dim:
                n = n + (_s,)
        n = n + dim
        x = x.permute(n)
        ns = x.shape[0 : -len(dim)] + (-1,)
        r = torch.max(x.reshape(ns), -1, keepdim=False)[0]
        rl = list(r.shape)
        for _sd in dim:
            rl.insert(_sd, 1)
        rf = tuple(rl)
        return (torch.reshape(r, rf),)

    return _impl, op.inputs[1:]


def pytorch_gather(op):
    """Implementation of gather for pytorch."""

    def _impl(x, dim, index):
        dim = dim.item()
        return (torch.gather(x, dim, index),)

    return _impl, op.inputs[1:]


def pytorch_scatter(op):
    """Implementation of scatter for pytorch."""

    def _impl(x, dim, index, src):
        dim = dim.item()
        return (torch.scatter(x, dim, index, src),)

    return _impl, op.inputs[1:]


def pytorch_scatter_add(op):
    """Implementation of scatter_add for pytorch."""

    def _impl(x, dim, index, src):
        dim = dim.item()
        return (torch.scatter_add(x, dim, index, src),)

    return _impl, op.inputs[1:]


def pytorch_concat(op):
    """Implementation of concat for pytorch."""

    def _impl(x, dim):
        dim = dim.item()
        return (torch.cat(x, dim),)

    return _impl, op.inputs[1:]


def pytorch_split(op):
    """Implementation of split for pytorch."""

    def _impl(x, sections, dim):
        dim = dim.item()
        return (torch.split(x, sections, dim),)

    return _impl, op.inputs[1:]


def pytorch_conv2d(op):
    """Implementation of conv2d for pytorch."""

    def _impl(input, weight, stride, padding, dilation, groups):
        groups = groups.item()
        return (
            torch.nn.functional.conv2d(
                input, weight, None, stride, padding, dilation, groups
            ),
        )

    return _impl, op.inputs[1:]


def pytorch_conv_transpose2d(op):
    """Implementation of conv_transpose2d."""

    def _impl(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        stride = tuple(_x.item() for _x in stride)
        padding = tuple(_x.item() for _x in padding)
        output_padding = tuple(_x.item() for _x in output_padding)
        dilation = tuple(_x.item() for _x in dilation)
        groups = groups.item()
        return torch.conv_transpose2d(
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )

    return _impl, op.inputs[1:]


def pytorch_conv2d_weight_grad(op):
    """Implementation of conv2d_weight_grad for pytorch."""

    def _impl(
        input, weight_size, grad_output, stride, padding, dilation, groups
    ):
        weight_size = tuple(w.item() for w in weight_size)
        stride = tuple(_x.item() for _x in stride)
        padding = tuple(_x.item() for _x in padding)
        dilation = tuple(_x.item() for _x in dilation)
        groups = groups.item()
        return (
            conv2d_weight(
                input,
                weight_size,
                grad_output,
                stride,
                padding,
                dilation,
                groups,
            ),
        )

    return _impl, op.inputs[1:]


def pytorch_max_pool2d(op):
    """Implementation of max_pool2d for pytorch."""

    def _impl(input, kernel_size, stride, padding, dilation, ceil_mode):
        return (
            torch.nn.functional.max_pool2d(
                input,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode.item(),
                False,
            ),
        )

    return _impl, op.inputs[1:]


def pytorch_max_pool2d_grad(op):
    """Implementation of max_pool2d grad for pytorch."""

    def _impl(input, kernel_size, stride, padding, dilation, ceil_mode, dout):
        input.requires_grad_(requires_grad=True)
        output = torch.nn.functional.max_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode.item(),
            False,
        )
        grads = torch.autograd.grad(output, input, dout, allow_unused=True)
        return (grads[0],)

    return _impl, op.inputs[1:]


_mapping = {
    P.array_cast: pytorch_array_cast,
    P.array_map: pytorch_array_map,
    P.array_reduce: pytorch_array_reduce,
    P.array_getitem: pytorch_array_getitem,
    P.array_setitem: pytorch_array_setitem,
    P.concat: pytorch_concat,
    P.conv2d: pytorch_conv2d,
    P.conv_transpose2d: pytorch_conv_transpose2d,
    P.conv2d_weight_grad: pytorch_conv2d_weight_grad,
    P.scalar_cast: pytorch_scalar_cast,
    P.max_pool2d: pytorch_max_pool2d,
    P.max_pool2d_grad: pytorch_max_pool2d_grad,
    P.gather: pytorch_gather,
    P.scatter: pytorch_scatter,
    P.scatter_add: pytorch_scatter_add,
    P.split: pytorch_split,
    P.argmax: pytorch_argmax,
    P.array_max: pytorch_array_max,
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

        :device: the target device for data storage ('cpu', 'cuda', 'cuda:X')

    """

    def __init__(self, device):
        """Create a PyTorch backend on the given device."""
        self.device = torch.device(device)
        self.compiler = CompileGraphs(
            lambda lst: pytorch_convert(lst, self), nonlinear_ops, self
        )

    def compile(self, graph, *others):
        """Compile a graph."""
        manage(graph)
        graph = closure_convert(graph)
        return self.compiler.compile_and_link(graph)

    def to_numpy(self, v):
        """Make a numpy array from a torch tensor."""
        if v.is_cuda:
            with untested_legacy():
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

    def from_backend_value(self, v, t):
        """Convert a backend value to an intermediate value."""
        if isinstance(t, abstract.AbstractScalar):
            return self.to_scalar(v)
        elif isinstance(t, abstract.AbstractArray):
            # Convert torch tensor to numpy tensor.
            output = self.to_numpy(v)
            # If possible and necessary, cast numpy tensor to expected tensor.
            array_type = t.element.xtype()
            if array_type and array_type not in _type_map:
                # Probably u16, u32 or u64. Let's cast.
                output = output.astype(type_to_np_dtype(array_type))
            return output
        elif isinstance(t, abstract.AbstractTuple):
            return tuple(
                self.from_backend_value(ve, te) for ve, te in zip(v, t.elements)
            )
        elif isinstance(t, abstract.AbstractTaggedUnion):
            return TaggedValue(
                v.tag, self.from_backend_value(v.value, t.options.get(v.tag))
            )
        elif isinstance(t, abstract.AbstractRandomState):
            return RandomStateWrapper(self.to_numpy(v))
        else:
            raise NotImplementedError(f"Don't know what to do for {t}")

    def to_backend_value(self, v, t):
        """Convert an intermediate value to a backend value."""
        if isinstance(t, abstract.AbstractError) or v is abstract.DEAD:
            return None
        elif isinstance(t, abstract.AbstractType):
            # Handle abstract types.
            # Return None if type does not match any torch type.
            myia_type = t.element.xtype()
            return _type_map.get(myia_type, None)
        elif isinstance(t, abstract.AbstractArray):
            return self.from_numpy(v)
        elif isinstance(t, abstract.AbstractScalar):
            if issubclass(
                t.values[abstract.TYPE], (xtype.Number, xtype.Bool, xtype.Nil)
            ):
                return self.from_scalar(v, t.values[abstract.TYPE])
            elif issubclass(t.values[abstract.TYPE], xtype.EnvType):
                assert len(v._contents) == 0
                return ()
            else:
                raise NotImplementedError(f"to_backend_value for {t}")
        elif isinstance(t, abstract.AbstractTuple):
            return tuple(
                self.to_backend_value(v, t) for v, t in zip(v, t.elements)
            )
        elif isinstance(t, abstract.AbstractTaggedUnion):
            real_t = t.options.get(v.tag)
            return TaggedValue(v.tag, self.to_backend_value(v.value, real_t))
        else:
            raise NotImplementedError(f"to_backend_value for {t}")


def PyTorchBackendR(device):
    """Pytorch proxy."""
    return HandleBackend(PyTorchBackend(device))


__all__ = ["PyTorchBackend", "PyTorchBackendR"]
