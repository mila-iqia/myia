"""Implementation of the 'is_not' operation."""

import operator

from ..lib import core


@core
def is_not(x, y):
    """Implementation of the `is not` operator."""
    return not (x is y)


__operation_defaults__ = {
    'name': 'is_not',
    'registered_name': 'is_not',
    'mapping': is_not,
    'python_implementation': operator.is_not,
}
