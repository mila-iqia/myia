"""Implementation of operation for numpy.full."""

import numpy as np

from ..lib import (
    abstract_array,
    core,
    distribute,
    myia_to_array,
    scalar_cast,
    to_scalar_type,
    typeof,
)


def pyimpl_full(shape, fill_value, dtype):
    """Python implementation for operation full."""
    return np.full(shape, fill_value, dtype)


@core
def full(shape, fill_value, dtype=None):
    """Main code for operation full.

    Arguments:
        shape: a tuple of integers
        fill_value: a scalar value
        dtype: either a string (e.g. 'int32')
            or a numpy dtype (e.g. np.int32)

    Returns:
        an array

    """
    if dtype is None:
        dtype = typeof(fill_value)
    abstract_scalar_type = to_scalar_type(dtype)
    scalar_value = scalar_cast(fill_value, abstract_scalar_type)
    return distribute(
        myia_to_array(scalar_value, abstract_array(shape, scalar_value)),
        shape
    )


__operation_defaults__ = {
    'name': 'full',
    'registered_name': 'full',
    'mapping': full,
    'python_implementation': pyimpl_full,
}
