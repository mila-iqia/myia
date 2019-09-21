"""Map the 'slice' operation."""

from ..classes import Slice

__operation_defaults__ = {
    'name': 'slice',
    'registered_name': 'slice',
    'mapping': Slice,
    'python_implementation': slice,
}
