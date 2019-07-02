"""PyTorch Frontend."""

import torch

from ..abstract.infer import to_abstract
from ..abstract.data import AbstractClassBase, AbstractArray, AbstractScalar, \
    ANYTHING, VALUE, TYPE, SHAPE
from ..dtype import Int, UInt, Float, Bool
from ..utils import overload

from . import Frontend

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

class AbstractModule(AbstractClassBase):
    """Represents a PyTorch Module."""

class AbstractTensor(AbstractArray):
    """Represents a PyTorch Tensor."""

    def __init__(self, element, values, requires_grad, retain_grad=None):
        """Initialize an AbstractValue."""
        super().__init__(element, values)
        self.requires_grad = requires_grad

        #TODO: IS RETAIN_GRAD EVEN NEEDED? IDK YET.
        self.retain_grad = retain_grad

class AbstractParameter(AbstractTensor):
    """Represents a PyTorch Parameter."""

    #TODO: Something to log that this is a Tensor to be updated when backward is called.


@to_abstract.variant
def _to_abstract(self, v: torch.nn.Module, context, ref, loop):
    fwd_fn = getattr(type(v), 'forward')
    
    attrs = {c[0]: self(c[1]) for c in v._modules.items()}

    for par_k, par_v in v._parameters.items():
        attrs[par_k] = self(par_v)

    for var_k, var_v in vars(v).items():
        #TODO: should torch.nn.Module be an instance type to check for?
        if isinstance(var_v, (torch.nn.Module, torch.Tensor)):
            attrs[var_k] = self(var_v)

    return AbstractModule(v.__class__, attrs, {'__call__': fwd_fn})


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
    return AbstractParameter(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: pytorch_dtype_to_type(v.dtype),
        }),
        {SHAPE: tuple(v.shape)},
        v.requires_grad,
        v.retain_grad
    )


class PyTorchFrontend(Frontend):
    """Frontend to run using PyTorch.

    Frontend options:
        

    """
    def __init__(self):
        pass

    to_abstract = staticmethod(_to_abstract)
