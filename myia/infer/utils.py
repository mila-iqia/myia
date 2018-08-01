"""Misc utilities for inference."""


from ..utils import Named


# Represents an unknown value
ANYTHING = Named('ANYTHING')


class InferenceError(Exception):
    """Inference error in a Myia program.

    Attributes:
        message: The error message.
        refs: A list of references which are involved in the error,
            e.g. because they have the wrong type or don't match
            each other.
        traceback_refs: A map from a context to the first reference in
            that context that fails to resolve because of this error.
            This represents a traceback of sorts.

    """

    def __init__(self, message, refs=[], app=None):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.traceback_refs = {}
        if app is not None:
            self.traceback_refs[app.context] = app


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""


class ValueWrapper:
    """Wrapper for an inferred value.

    Values may be wrapped using subclasses of ValueWrapper, associating them
    with tracking data or metadata.
    """

    def __init__(self, value):
        """Initialize a ValueWrapper."""
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return type(other) is type(self) \
            and self.value == other.value

    @property
    def __unwrapped__(self):
        return self.value
