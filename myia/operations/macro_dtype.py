"""Implementation of the 'dtype' macro."""

from ..lib import AbstractArray, Constant, macro


@macro
async def dtype(info, arr: AbstractArray):
    """Macro implementation for 'dtype'."""
    return Constant((await arr.get()).element)


__operation_defaults__ = {
    "name": "dtype",
    "registered_name": "dtype",
    "mapping": dtype,
    "python_implementation": None,
}
