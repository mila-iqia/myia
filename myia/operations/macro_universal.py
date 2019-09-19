"""Implementation of the 'universal' macro."""

from .. import lib, operations, xtype
from ..lib import MyiaTypeError, macro


@macro
async def universal(info, fn):
    """Macro implementation for 'universal'."""
    raise NotImplementedError()


__operation_defaults__ = {
    'name': 'universal',
    'registered_name': 'universal',
    'mapping': universal,
    'python_implementation': None,
}
