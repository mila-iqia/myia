"""Utilities for operations and primitives."""


from ..utils import HasDefaults


class Primitive(HasDefaults):
    """Base class for primitives."""

    def __init__(self, name, defaults={}):
        """Initialize a Primitive."""
        super().__init__(name, defaults, "__primitive_defaults__")

    @property
    def universal(self):
        """Whether this primitive is universal or not."""
        return self.defaults().get("universal", False)

    def __str__(self):
        return self.name

    __repr__ = __str__


class InferencePrimitive(Primitive):
    """Represents a primitive that is eliminated after inference."""


class PlaceholderPrimitive(Primitive):
    """Represents a primitive that is eliminated during optimization."""


class BackendPrimitive(Primitive):
    """Represents a primitive that must be implemented by the backend."""


class Operation(HasDefaults):
    """Represents an operation."""

    def __init__(self, name, defaults={}):
        """Initialize an Operation."""
        super().__init__(name, defaults, "__operation_defaults__")

    def __call__(self, *args, **kwargs):
        """Call an Operation."""
        dflt = self.defaults()
        impl = dflt.get("python_implementation", None)
        if impl is None:
            raise RuntimeError(
                f"Myia-only operation {self.name} cannot be called directly."
            )
        else:
            return impl(*args, **kwargs)

    def __str__(self):
        return f"myia.operations.{self.name}"

    __repr__ = __str__


class OperationDefinition(dict):
    """Definition of an operation."""

    def __init__(self, **kwargs):
        """Initialize an OperationDefinition."""
        assert "name" in kwargs
        super().__init__(**kwargs)

    def __to_myia__(self):
        """Return the mapping."""
        return self["mapping"]

    def __call__(self, *args, **kwargs):
        """Raise an error when calling the OperationDefinition."""
        raise TypeError(
            f'Operation definition for {self["name"]} is not callable.'
        )


def to_opdef(fn):
    """Create an operation definition from a function."""
    name = fn.__name__
    return OperationDefinition(
        name=name, registered_name=name, mapping=fn, python_implementation=None
    )
