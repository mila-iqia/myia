"""Implementation of the 'embed' operation."""

from ..lib import Constant, SymbolicKeyInstance, macro, sensitivity_transform


@macro
async def embed(info, x):
    """Return a constant that embeds the identity of the input node."""
    typ = sensitivity_transform(await x.get())
    key = SymbolicKeyInstance(x.node, typ)
    return Constant(key)


__operation_defaults__ = {
    'name': 'embed',
    'registered_name': 'embed',
    'mapping': embed,
    'python_implementation': None,
}
