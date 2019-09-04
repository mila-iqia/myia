"""PyTorch Frontend."""

import copy
from collections import OrderedDict

import torch
import torch.utils.dlpack

from .. import composite as C, macros as M
from ..abstract.data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
)
from ..abstract.infer import to_abstract
from ..composite import core
from ..dtype import Bool, Float, Int, Number, UInt
from ..hypermap import hyper_map
from ..opt.clean import _reabs
from ..pipeline.resources import standard_method_map, standard_object_map
from ..pipeline.steps import convert_arg, convert_result_array
from ..prim import ops as P
from ..prim.py_implementations import scalar_cast, scalar_to_array
from .pytorch_abstract_types import (
    APT,
    AbstractModule,
    AbstractPyTorchTensor,
    PyTorchTensorWrapper,
)
from .pytorch_functions import _sum, conv2d, item, linear, relu, sigmoid

_type_map = {
    torch.int8: Int[8],
    torch.int16: Int[16],
    torch.int32: Int[32],
    torch.int64: Int[64],
    torch.uint8: UInt[8],
    torch.float16: Float[16],
    torch.float32: Float[32],
    torch.float64: Float[64],
    torch.uint8: Bool,
}


def pytorch_dtype_to_type(dtype):
    """Map a pytorch dtype to a myia type."""
    if dtype not in _type_map:
        raise TypeError(f"Unsupported dtype {dtype}")
    return _type_map[dtype]


standard_object_map.update({
    torch.exp: C.exp,
    torch.log: C.log,
    torch.mm: P.dot,
    torch.relu: relu,
    torch.reshape: P.reshape,
    torch.sigmoid: sigmoid,
    torch.sum: _sum,
    torch.t: C.transpose,
    torch.tanh: C.tanh,
    # torch.zeros_like: C.zeros_like,  # currently only works with pt backend
    torch.nn.functional.linear: linear,
    torch.nn.functional.conv2d: conv2d,
})


C.exp.register(APT)(C.array_exp)
C.log.register(APT)(C.array_log)
C.tanh.register(APT)(C.array_tanh)


@C._leaf_zeros_like.register(APT)
@core
def _array_zero(xs):
    scalar_zero = P.scalar_cast(0, C.typeof(xs).element)
    return P.distribute(C.to_array(scalar_zero, C.typeof(xs)),
                        P.shape(xs))


standard_method_map[AbstractPyTorchTensor] = \
    standard_method_map[AbstractArray].copy()
standard_method_map[AbstractPyTorchTensor].update({
    'dim': C.ndim,
    'exp': C.exp,
    'item': item,
    'log': C.log,
    'relu': relu,
    'reshape': P.reshape,
    'sigmoid': sigmoid,
    'shape': property(P.shape),
    'sum': _sum,
    't': C.transpose,
    'tanh': C.tanh,
    'view': P.reshape,  # contiguousness is ignored by us for now?
    'zeros_like': C.zeros_like,  # hidden method used by bwd (I think)
})


# TODO: mod_* for other arithmetic besides sub
@core
def mod_sub(self, x):
    """Hypermap subtraction (used for subtracting modules during update)."""
    return hyper_map(C.sub, self, x)

##############################################################################


# # This might end up as an alternative to blacklist of Module constructors.
# # I.e. get the list of constructors from a dummy pytorch module.
# class DummyModule(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#     def forward(self, x):
#        return 9

# dummy_module =  DummyModule()


# TODO: should all of these actually be blacklisted (not used).
# Curently blacklists all constructors except '_parameters' and '_modules'.
# 'training' should probably be removed from blacklist in next PR.
blacklist = ('_backend', '_buffers', '_backward_hooks', '_forward_hooks',
             '_forward_pre_hooks', '_state_dict_hooks',
             '_load_state_dict_pre_hooks',

             'training'
             )


@to_abstract.register
def _to_abstract(self, v: torch.nn.Module, **kwargs):
    fwd_fn = getattr(type(v), 'forward')
    attrs = {}
    for var_k, var_v in vars(v).items():
        if var_k not in blacklist:

            # TODO: Remove "(isinstance(v, torch.nn.Sequential) and"
            #       once Alias PR ready
            # TODO: Remove rest of if statement once Dict supports empty Dict
            if var_k not in ('_parameters', '_modules') or \
                    (isinstance(v, torch.nn.Sequential) and
                     var_v != OrderedDict()):
                attrs[var_k] = self(var_v, **kwargs)
        else:
            pass
            # TODO: maybe make a warning for if user happened
            #       to name attribute something in blacklist

    # TODO: Remove "if not isinstance(v, Sequential)" once Alias PR is ready
    # """TODO: Remove these 2 loops (mod and par) once Dict support empty Dict
    if not isinstance(v, torch.nn.Sequential):
        for mod_k, mod_v in v._modules.items():
            attrs[mod_k] = self(mod_v, **kwargs)

        for par_k, par_v in v._parameters.items():
            attrs[par_k] = self(par_v, **kwargs)

    # TODO: figure out how to delattr so that memory doesn't double
    # for k in attrs:
    #     delattr(v, k)

    names = list(attrs.keys())

    def new_module(*args):
        nonlocal v
        # TODO: Figure out something more memory efficient than deepcopy.
        #       P.S. We tried copy.copy(v) and it is not sufficiently deep.
        v = copy.deepcopy(v)
        for k, a in zip(names, args):
            if isinstance(getattr(v, k), torch.nn.Parameter):
                setattr(v, k, torch.nn.Parameter(a))
            else:
                setattr(v, k, a)
        return v

    return AbstractModule(v.__class__, attrs, {'__call__': fwd_fn,
                          '__sub__': mod_sub}, constructor=new_module)


@to_abstract.register  # noqa: F811
def _to_abstract(self, v: torch.Tensor, **kwargs):
    return AbstractPyTorchTensor(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        # v.requires_grad,
        # v.retain_grad
    )


@to_abstract.register  # noqa: F811
def _to_abstract(self, v: torch.nn.Parameter, **kwargs):
    # return AbstractPyTorchParameter(
    return AbstractPyTorchTensor(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        # v.requires_grad,
        # v.retain_grad
    )


@to_abstract.register  # noqa: F811
def _to_abstract(self, v: PyTorchTensorWrapper, **kwargs):
    return AbstractPyTorchTensor(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        # v.requires_grad,
        # v.retain_grad
    )

##############################################################################


@convert_arg.register
def _convert_arg(self, arg, orig_t: AbstractPyTorchTensor):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.values[TYPE]
    assert issubclass(et, Number)
    if isinstance(arg, torch.Tensor):
        arg = arg.detach().numpy()
    return arg

##############################################################################


@convert_result_array.register
def _convert_result_array(arg, orig_t: AbstractPyTorchTensor):
    return torch.from_numpy(arg)

##############################################################################


@M._cast_helper.register(Number, APT)
@core
def _pt__cast_helper(x, model):
    t = P.typeof(model)
    return scalar_to_array(scalar_cast(x, t.element), P.typeof(model))


@_reabs.register
def _pt__reabs(self, a: AbstractPyTorchTensor):
    return (yield AbstractArray)(self(a.element), a.values)
