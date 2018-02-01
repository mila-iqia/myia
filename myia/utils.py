"""General utilities and design patterns."""
from typing import Any, List, Dict, TypeVar


T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Named:
    """A named object.

    This class can be used to construct objects with a name that will be used
    for the string representation.

    """

    def __init__(self, name):
        """Construct a named object.

        Args:
            name: The name of this object.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


class Registry(Dict[T1, T2]):
    """Associates primitives to implementations."""

    def __init__(self) -> None:
        """Initialize a Registry."""
        super().__init__()

    def register(self, prim):
        """Register a primitive."""
        def deco(fn):
            """Decorate the function."""
            self[prim] = fn
            return fn
        return deco


def repr_(obj: Any, **kwargs: Any):
    """Return unique string representation of object with additional info.

    The usual representation is `<module.Class object at address>`. This
    function returns `<module.Class(key=value) object at address>` instead, to
    make objects easier to identify by their attributes.

    Args:
        **kwargs: The attributes and their values that will be printed as part
            of the string representation.

    """
    name = f'{obj.__module__}.{obj.__class__.__name__}'
    info = ', '.join(f'{key}={value}' for key, value in kwargs.items())
    address = str(hex(id(obj)))
    return f'<{name}({info}) object at {address}>'


def list_str(lst: List):
    """Return string representation of a list.

    Unlike the default string representation, this calls `str` instead of
    `repr` on each element.

    """
    elements = ', '.join(str(elem) for elem in lst)
    return f'[{elements}]'
