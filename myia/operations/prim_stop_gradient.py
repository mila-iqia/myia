"""
Definition of primitive `stop_gradient`.

Used to stop gradient propagation through given input.
"""

from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import zeros_like
from . import primitives as P


def pyimpl_stop_gradient(x):
    """Implement `stop_gradient`."""
    return x


@standard_prim(P.stop_gradient)
async def infer_stop_gradient(self, engine, x):
    """Infer the return type of primitive `stop_gradient`."""
    return x


@bprop_to_grad_transform(P.stop_gradient)
def bprop_stop_gradient(x, out, dout):
    """Backpropagator for primitive `stop_gradient`."""
    return (zeros_like(x),)


__operation_defaults__ = {
    'name': 'stop_gradient',
    'registered_name': 'stop_gradient',
    'mapping': P.stop_gradient,
    'python_implementation': pyimpl_stop_gradient,
}


__primitive_defaults__ = {
    'name': 'stop_gradient',
    'registered_name': 'stop_gradient',
    'type': 'placeholder',
    'python_implementation': pyimpl_stop_gradient,
    'inferrer_constructor': infer_stop_gradient,
    'grad_transform': bprop_stop_gradient,
}
