"""PyTorch Frontend."""

import torch
import torch.utils.dlpack

from .. import composite as C
from ..prim import ops as P
from ..abstract.infer import to_abstract, ArrayWrapper
from ..abstract.data import AbstractClassBase, AbstractArray, AbstractScalar, \
    ANYTHING, VALUE, TYPE, SHAPE
from ..abstract.utils import pytype_to_abstract
from ..dtype import Int, UInt, Float, Bool, Nil, Number
from ..utils import overload
from ..pipeline.resources import standard_object_map, standard_method_map
from ..hypermap import hyper_map

from . import Frontend
from .pytorch_functions import linear, tensor_dim, t

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
    # This is a hack but we really need uint64 support
    # torch.int64: Int[64],
}

def pytorch_dtype_to_type(dtype):
    """Map a pytorch dtype to a myia type."""
    if dtype not in _type_map:
        raise TypeError(f"Unsupported dtype {dtype}")
    return _type_map[dtype]

'''
def pytorch_type_to_myia_type(t):
    """Map myia types to pytorch types."""
    if t not in _type_map:  # pragma: no cover
        raise TypeError(f"Unsupported type: {t}")
    return _type_map[t]
    '''

pytorch_object_map = standard_object_map.copy()
pytorch_object_map.update({
    torch.nn.functional.linear: linear,
    torch.tanh: C.tanh,
    })



class AbstractModule(AbstractClassBase):
    """Represents a PyTorch Module."""

    def user_defined_version(self):
        return self

class AbstractTensor(AbstractArray):
    """Represents a PyTorch Tensor."""

    #def __init__(self, element, values, requires_grad, retain_grad=None):
    def __init__(self, element, values, requires_grad=None, retain_grad=None):    
        """Initialize an AbstractValue."""
        super().__init__(element, values)
        self.requires_grad = requires_grad

        #TODO: IS RETAIN_GRAD EVEN NEEDED? IDK YET.
        self.retain_grad = retain_grad

class AbstractParameter(AbstractTensor):
    """Represents a PyTorch Parameter."""

    #TODO: Something to log that this is a Tensor to be updated when backward is called.


AT = AbstractTensor(ANYTHING, {SHAPE: ANYTHING})

@C.tanh.register(AT)
@C.core
def array_tanh(xs):
    """Implementation of `array_tanh`."""
    return P.array_map(P.scalar_tanh, xs)

'''
@C.tanh.register(AT)
@C.core
def array_tanh(xs):
    """Implementation of `array_tanh`."""
    return P.array_map(P.scalar_tanh, xs)
    #'''


@C._leaf_zeros_like.register(AT)
@C.core
def _array_zero(xs):
    scalar_zero = P.scalar_cast(0, C.typeof(xs).element)
    return P.distribute(C.to_array(scalar_zero), P.shape(xs))


pytorch_method_map = standard_method_map.copy()
pytorch_method_map[AbstractTensor] = pytorch_method_map[AbstractArray].copy()
pytorch_method_map[AbstractTensor].update({
    'dim': tensor_dim,
    't': t,
    'tanh': C.tanh,
    'zeros_like': C.zeros_like,
    })

"""
dummy_module = class DummyModule(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
       return 99
       #"""


# TODO: should all of these actually be blacklisted (not used)
blacklist = ('_backend', '_buffers', '_backward_hooks', '_forward_hooks',
             '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
             
             'training')


@C.core
def mod_sub(self, x):
        return hyper_map(C.sub, self, x)


@to_abstract.variant
def _to_abstract(self, v: torch.nn.Module, context, ref, loop):
    fwd_fn = getattr(type(v), 'forward')
    attrs = {}
    for var_k, var_v in vars(v).items():
        if var_k not in blacklist:

            #TODO: Remove this if statement line once Dict PR is ready
            if var_k not in ('_parameters', '_modules'):

                attrs[var_k] = self(var_v)
        else:
            pass
            #TODO: maybe make a warning for if user happened 
            #      to name attribute something in blacklist

    #""" TODO: remove these 2 loops (module and params) once Dict PR is ready
    for mod_k, mod_v in v._modules.items():
        attrs[mod_k] = self(mod_v)

    for par_k, par_v in v._parameters.items():
        attrs[par_k] = self(par_v)
        #"""

    #TODO: figure out how to delattr so that memory doesn't double
    #for k in attrs:
    #    delattr(v, k)

    names = list(attrs.keys())

    def new_module(*args):
        for k, a in zip(names, args):
            #TODO: make more robust (e.g. if relay backend is used)
            if isinstance(getattr(v, k), torch.nn.Parameter):
                setattr(v, k, torch.nn.Parameter(a))
            else:
                setattr(v, k, a)
        return v

    return AbstractModule(v.__class__, attrs, {'__call__': fwd_fn,
        '__sub__': mod_sub}, constructor=new_module)


@overload
def _to_abstract(self, v: torch.Tensor, context, ref, loop):
    return AbstractTensor(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        v.requires_grad,
        v.retain_grad
    )


@overload
def _to_abstract(self, v: torch.nn.Parameter, context, ref, loop):
    #return AbstractParameter(
    return AbstractTensor(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        v.requires_grad,
        v.retain_grad
    )


@overload
def _to_abstract(self, v: torch.nn.backends.thnn.THNNFunctionBackend, context, ref, loop):
    return AbstractScalar({
            VALUE: ANYTHING,
            TYPE: Nil,
        })


from ..pipeline.steps import convert_arg
@convert_arg.variant
def _convert_arg(self, arg, orig_t: AbstractArray, backend):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.values[TYPE]
    assert issubclass(et, Number)
    if isinstance(arg, ArrayWrapper):
        arg = arg.array
    if isinstance(arg, torch.Tensor):
        arg = backend.from_dlpack(torch.utils.dlpack.to_dlpack(arg))
    backend.check_array(arg, et)
    return arg


#"""
from ..pipeline.steps import convert_result
@convert_result.variant
def _convert_result(self, arg, orig_t, vm_t: AbstractArray, backend,
                   return_backend):
    if not isinstance(arg, torch.Tensor):
        arg = torch.utils.dlpack.from_dlpack(backend.to_dlpack(arg))
    return arg


# def _convert_result(self, res, orig_t, vm_t: torch.nn.Module, backend,
#                    return_backend):
#     oe = orig_t.attributes.values()
#     ve = vm_t.attributes.values()
#     tup = tuple(self(getattr(res, attr), o, v, backend, return_backend)
#                 for attr, o, v in zip(orig_t.attributes, oe, ve))
#     return orig_t.tag(*tup)
    #"""


class PyTorchFrontend(Frontend):
    """Frontend to run using PyTorch.

    Frontend options:
        

    """
    def __init__(self):
        pass

    to_abstract = staticmethod(_to_abstract)
    convert_arg = staticmethod(_convert_arg)
    convert_result = staticmethod(_convert_result)

    def configure(self, pip):
        return pip.configure({'convert.object_map': pytorch_object_map,
                'method_map': pytorch_method_map,
                'array_class': AbstractTensor})
