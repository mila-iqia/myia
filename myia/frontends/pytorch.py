"""PyTorch Frontend."""

import copy
import types
from collections import OrderedDict

import torch

from .. import operations
from ..abstract.data import (
    ALIASID,
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
)
from ..abstract.infer import to_abstract
from ..classes import ADT
from ..hypermap import hyper_map
from ..operations import primitives as P
from ..pipeline.resources import default_convert
from ..pipeline.standard import standard_method_map, standard_object_map
from ..utils import OrderedSet, core, get_fields
from ..xtype import NDArray
from .pytorch_abstract_types import (
    AbstractModule,
    PyTorchTensor,
    pytorch_dtype_to_type,
)
from .pytorch_functions import (
    _max,
    _sum,
    argmax,
    cat,
    conv2d,
    gather,
    item,
    linear,
    log_softmax,
    max_pool2d,
    mean,
    nll_loss,
    relu,
    reshape,
    scatter,
    scatter_add,
    sigmoid,
    size,
    smooth_l1_loss,
    softmax,
    split,
    squeeze,
    stack,
    std,
    transpose,
    unsqueeze,
    var,
    view_as,
    zeros,
)

standard_object_map.update({
    torch.abs: operations.array_abs,
    torch.argmax: argmax,
    torch.cat: cat,
    torch.eq: operations.array_eq,
    torch.exp: operations.array_exp,
    torch.gather: gather,
    torch.log: operations.array_log,
    torch.log_softmax: log_softmax,
    torch.max: _max,
    torch.mean: mean,
    torch.mm: P.dot,
    torch.pow: operations.array_pow,
    torch.relu: relu,
    torch.reshape: reshape,
    torch.scatter: scatter,
    torch.scatter_add: scatter_add,
    torch.sigmoid: sigmoid,
    torch.sign: operations.array_sign,
    torch.softmax: softmax,
    torch.split: split,
    torch.squeeze: squeeze,
    torch.stack: stack,
    torch.std: std,
    torch.sum: _sum,
    torch.t: operations.t,
    torch.tanh: operations.array_tanh,
    torch.transpose: transpose,
    torch.unsqueeze: unsqueeze,
    torch.var: var,
    # torch.zeros_like: C.zeros_like,  # currently only works with pt backend
    torch.nn.functional.conv2d: conv2d,
    torch.nn.functional.linear: linear,
    torch.nn.functional.max_pool2d: max_pool2d,
    torch.nn.functional.nll_loss: nll_loss,
    torch.nn.functional.smooth_l1_loss: smooth_l1_loss,

    torch.zeros: zeros,
})


standard_method_map[PyTorchTensor] = \
    standard_method_map[NDArray].copy()
standard_method_map[PyTorchTensor].update({
    'abs': operations.array_abs,
    'dim': operations.ndim,
    'dtype': property(operations.dtype),
    'argmax': argmax,
    'eq': operations.array_eq,
    'exp': operations.array_exp,
    'gather': gather,
    'item': item,
    'log': operations.array_log,
    'log_softmax': log_softmax,
    'max': _max,
    'mean': mean,
    'permute': P.transpose,
    'pow': operations.array_pow,
    'relu': relu,
    'reshape': reshape,
    'scatter': scatter,
    'scatter_add': scatter_add,
    'sigmoid': sigmoid,
    'shape': property(operations.shape),
    'sign': operations.array_sign,
    'size': size,
    'softmax': softmax,
    'split': split,
    'squeeze': squeeze,
    'std': std,
    'sum': _sum,
    't': operations.t,
    'transpose': transpose,
    'tanh': operations.array_tanh,
    'unsqueeze': unsqueeze,
    'var': var,
    'view': reshape,  # contiguousness is ignored by us for now?
    'view_as': view_as,  # contiguousness is ignored by us for now?
    'zeros_like': operations.zeros_like,  # hidden method used by bwd (I think)
})


# TODO: mod_* for other arithmetic besides sub
@core
def mod_sub(self, x):
    """Hypermap subtraction (used for subtracting modules during update)."""
    return hyper_map(operations.sub, self, x)

##############################################################################


blacklist = set(dir(torch.nn.Module()))
blacklist.add('__constants__')
blacklist.add('reset_parameters')


@to_abstract.register
def _to_abstract(self, v: torch.nn.Module, **kwargs):
    standard_method_map[type(v)] = {
        '__call__': getattr(type(v), 'forward'),
        '__sub__': mod_sub,
    }
    fields = {}
    for var_k in dir(v):
        if (var_k not in blacklist) or (var_k in ('_parameters', '_modules')):
            var_v = getattr(v, var_k)
            if not isinstance(var_v, types.MethodType):
                # TODO: Remove "(isinstance(v, torch.nn.Sequential) and"
                #       once Alias PR ready
                # TODO: Remove rest of if statement once Dict support empty Dic
                if var_k not in ('_parameters', '_modules') or \
                        (isinstance(v, torch.nn.Sequential) and
                         var_v != OrderedDict()):

                    fields[var_k] = self(var_v, **kwargs)
        else:
            pass
            # TODO: maybe make a warning for if user happened
            #       to name attribute something in blacklist

    # TODO: Remove "if not isinstance(v, Sequential)" once Alias PR is ready
    # """TODO: Remove these 2 loops (mod and par) once Dict support empty Dict
    if not isinstance(v, torch.nn.Sequential):
        for mod_k, mod_v in v._modules.items():
            fields[mod_k] = self(mod_v, **kwargs)

        for par_k, par_v in v._parameters.items():
            fields[par_k] = self(par_v, **kwargs)
        # """

    # TODO: figure out how to delattr so that memory doesn't double
    # for k in fields:
    #     delattr(v, k)
    # for k in methods:
    #     delattr(type(v), k)

    names = list(fields.keys())

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

    return AbstractModule(v.__class__, fields, constructor=new_module)


@to_abstract.register  # noqa: F811
def _to_abstract(self, v: torch.Tensor, **kwargs):
    return AbstractArray(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape), TYPE: PyTorchTensor},
    )


@to_abstract.register  # noqa: F811
def _to_abstract(self, v: torch.nn.Parameter, alias_map={}, **kwargs):
    tracks = {SHAPE: tuple(v.shape), TYPE: PyTorchTensor}
    if id(v) in alias_map:
        tracks[ALIASID] = alias_map[id(v)]
    return AbstractArray(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        tracks,
    )


@default_convert.register  # noqa: F811
def _default_convert(env, x: torch.dtype):
    return default_convert(env, pytorch_dtype_to_type(x))


__all__ = [
    'pytorch_dtype_to_type',
]


@get_fields.register
def _get_fields(instance: torch.nn.Module):
    blacklist = OrderedSet(dir(torch.nn.Module()))
    blacklist.add('__constants__')
    blacklist.add('reset_parameters')

    blacklist.remove('_parameters')
    blacklist.remove('_modules')

    keys = OrderedSet(dir(instance)) - blacklist
    d = {}

    for k in keys:
        d[k] = getattr(instance, k)
    return d


def tensor_pytorch_aliasable(v, vseq, path):
    """Aliasing policy whereas all pytorch tensors are aliasable.

    Tensors inside a list or ADT are not aliasable.
    """
    if isinstance(v, torch.Tensor):
        if any(isinstance(x, (list, ADT)) for x in vseq):
            return 'X'
        else:
            return True
    return False
