"""Implementation of the 'print' operation."""

from ..lib import HandleInstance, core
from ..operations import ioprint, universe_getitem, universe_setitem

print_handle = HandleInstance(0)


@core(universal=True)
def print(U, *entries):
    """Implementation for 'print'."""
    for entry in entries:
        iostate0 = universe_getitem(U, print_handle)
        iostate = ioprint(iostate0, entry)
        U = universe_setitem(U, print_handle, iostate)
    return U, None


__operation_defaults__ = {
    'name': 'print',
    'registered_name': 'print',
    'mapping': print,
    'python_implementation': None,
}
