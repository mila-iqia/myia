"""
Definition of primitive `detach`.

Based on torch tensor method `detach()` (2019/12/03):
https://pytorch.org/docs/stable/tensors.html#torch.Tensor.detach
"""

from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import zeros_like
from . import primitives as P


def pyimpl_detach(x):
    """Implement `detach`."""
    return x


@standard_prim(P.detach)
async def infer_detach(self, engine, x):
    """Infer the return type of primitive `detach`."""
    return x


@bprop_to_grad_transform(P.detach)
def bprop_detach(x, out, dout):
    """Backpropagator for primitive `detach`."""
    return (zeros_like(x),)


__operation_defaults__ = {
    'name': 'detach',
    'registered_name': 'detach',
    'mapping': P.detach,
    'python_implementation': pyimpl_detach,
}


__primitive_defaults__ = {
    'name': 'detach',
    'registered_name': 'detach',
    'type': 'backend',
    'python_implementation': pyimpl_detach,
    'inferrer_constructor': infer_detach,
    'grad_transform': bprop_detach,
}
