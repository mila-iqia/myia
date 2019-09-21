"""Implementation of the 'enumerate' operation."""

from ..lib import core


@core
def enumerate_(seq):
    """Myia implementation of the standard enumerate function."""
    return zip(range(len(seq)), seq)


__operation_defaults__ = {
    'name': 'enumerate',
    'registered_name': 'enumerate',
    'mapping': enumerate_,
    'python_implementation': enumerate,
}
