"""Definitions for the primitive `random_initialize`."""

import numpy as np

from .. import xtype
from ..lib import AbstractRandomState, standard_prim
from . import primitives as P


def pyimpl_random_initialize(seed):
    """Implement `random_initialize`."""
    return np.random.RandomState(seed)


@standard_prim(P.random_initialize)
async def infer_random_initialize(self, engine, seed: xtype.u32):
    """Infer the return type of primitive `random_initialize`."""
    return AbstractRandomState()


__operation_defaults__ = {
    'name': 'random_initialize',
    'registered_name': 'random_initialize',
    'mapping': P.random_initialize,
    'python_implementation': pyimpl_random_initialize,
}


__primitive_defaults__ = {
    'name': 'random_initialize',
    'registered_name': 'random_initialize',
    'type': 'backend',
    'python_implementation': pyimpl_random_initialize,
    'inferrer_constructor': infer_random_initialize,
    'grad_transform': None,
}
