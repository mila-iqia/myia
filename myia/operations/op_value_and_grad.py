"""Implementation of the 'value_and_grad' operation."""

from ..lib import core
from ..operations import grad


@core
def value_and_grad(*args, **kwargs):
    """Return the value of the function along with the gradient."""
    return grad(*args, **kwargs, return_value=True)


__operation_defaults__ = {
    'name': 'value_and_grad',
    'registered_name': 'value_and_grad',
    'mapping': value_and_grad,
    'python_implementation': None,
}
