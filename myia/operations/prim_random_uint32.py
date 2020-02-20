"""Definitions for the primitive `random_uint32`."""

import numpy as np

from .. import xtype
from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractRandomState,
    AbstractScalar,
    AbstractTuple,
    standard_prim,
)
from . import primitives as P


def pyimpl_random_uint32(rstate, shape):
    """Implement `random_uint32`."""
    # Store 2**32 in uint64.
    high = np.uint64(np.iinfo(np.uint32).max) + np.uint64(1)
    # Generate random integers in interval [0, 2**32) in uint64.
    value = rstate.randint(low=0, high=high, size=shape, dtype='uint64')
    # Cast value to uint32.
    output = value.astype(np.uint32)
    return rstate, output


@standard_prim(P.random_uint32)
async def infer_random_uint32(self, engine,
                              rstate: AbstractRandomState,
                              shape: AbstractTuple):
    """Infer the return type of primitive `random_uint32`."""
    output_scalar_type = AbstractScalar({VALUE: ANYTHING, TYPE: xtype.u32})
    output_shape = tuple(
        self.require_constant(e, argnum=f'"3:size[{edx}]"')
        for edx, e in enumerate(shape.elements))
    value_type = AbstractArray(output_scalar_type,
                               {SHAPE: output_shape, TYPE: xtype.NDArray})
    return AbstractTuple((AbstractRandomState(), value_type))


__operation_defaults__ = {
    'name': 'random_uint32',
    'registered_name': 'random_uint32',
    'mapping': P.random_uint32,
    'python_implementation': pyimpl_random_uint32,
}


__primitive_defaults__ = {
    'name': 'random_uint32',
    'registered_name': 'random_uint32',
    'type': 'backend',
    'python_implementation': pyimpl_random_uint32,
    'inferrer_constructor': infer_random_uint32,
    'grad_transform': None,
}
