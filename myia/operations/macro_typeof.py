"""Implementation of the 'typeof' operation."""

from ..lib import Constant, macro


@macro
async def typeof(info, data):
    """Return a constant with the type of the input."""
    return Constant(await data.get())


__operation_defaults__ = {
    'name': 'typeof',
    'registered_name': 'typeof',
    'mapping': typeof,
    'python_implementation': None,
}
