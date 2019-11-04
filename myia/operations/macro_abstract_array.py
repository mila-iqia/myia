"""
Implementation of the 'abstract_array' operation.

Create an AbstractArray using given shape and AbstractScalar.
Used in myia/operations/op_full.py
"""

from ..lib import SHAPE, TYPE, AbstractArray, Constant, macro
from ..xtype import NDArray


@macro
async def abstract_array(info, shape, dtype):
    """Return a constant with the type of the input."""
    array_shape = await shape.get()
    scalar_type = await dtype.get()
    output_shape = tuple(
        element.xvalue() for element in array_shape.children())
    return Constant(
        AbstractArray(scalar_type, {SHAPE: output_shape, TYPE: NDArray}))


__operation_defaults__ = {
    'name': 'abstract_array',
    'registered_name': 'abstract_array',
    'mapping': abstract_array,
    'python_implementation': None,
}
