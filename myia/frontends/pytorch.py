"""PyTorch Frontend."""

import copy
import types
from collections import OrderedDict

import torch

from .. import composite as C
from ..abstract.data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
)
from ..abstract.infer import to_abstract
from ..hypermap import hyper_map
from ..pipeline.standard import standard_method_map, standard_object_map
from ..prim import ops as P
from ..utils import core
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
    conv2d,
    gather,
    item,
    linear,
    log_softmax,
    max_pool2d,
    nll_loss,
    relu,
    reshape,
    scatter,
    scatter_add,
    sigmoid,
    softmax,
    squeeze,
    t,
    transpose,
    zeros,
)

standard_object_map.update({
    torch.argmax: argmax,
    torch.exp: C.exp,
    torch.gather: gather,
    torch.log: C.log,
    torch.log_softmax: log_softmax,
    torch.max: _max,
    torch.mm: P.dot,
    torch.relu: relu,
    torch.reshape: reshape,
    torch.scatter: scatter,
    torch.scatter_add: scatter_add,
    torch.sigmoid: sigmoid,
    torch.softmax: softmax,
    torch.squeeze: squeeze,
    torch.sum: _sum,
    torch.t: t,
    torch.tanh: C.tanh,
    torch.transpose: transpose,
    # torch.zeros_like: C.zeros_like,  # currently only works with pt backend
    torch.nn.functional.conv2d: conv2d,
    torch.nn.functional.linear: linear,
    torch.nn.functional.max_pool2d: max_pool2d,
    torch.nn.functional.nll_loss: nll_loss,

    torch.zeros: zeros,
})


standard_method_map[PyTorchTensor] = \
    standard_method_map[NDArray].copy()
standard_method_map[PyTorchTensor].update({
    'dim': C.ndim,
    'argmax': argmax,
    'exp': C.exp,
    'gather': gather,
    'item': item,
    'log': C.log,
    'log_softmax': log_softmax,
    'max': _max,
    'permute': P.transpose,
    'relu': relu,
    'reshape': reshape,
    'scatter': scatter,
    'scatter_add': scatter_add,
    'sigmoid': sigmoid,
    'shape': property(P.shape),
    'softmax': softmax,
    'squeeze': squeeze,
    'sum': _sum,
    't': t,
    'transpose': transpose,
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
"""
blacklist = ('_backend', '_buffers', '_backward_hooks', '_forward_hooks',
             '_forward_pre_hooks', '_state_dict_hooks',
             '_load_state_dict_pre_hooks',

             'training'
             )
             """
blacklist = set(dir(torch.nn.Module()) + ["__constants__"])
blacklist.add('reset_parameters')


@to_abstract.register
def _to_abstract(self, v: torch.nn.Module, **kwargs):
    from ..pipeline.resources import standard_method_map
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
def _to_abstract(self, v: torch.nn.Parameter, **kwargs):
    return AbstractArray(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape), TYPE: PyTorchTensor},
    )
