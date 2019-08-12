"""Exceptions that may be raised within Myia."""

from contextvars import ContextVar

infer_trace = ContextVar('infer_trace')
infer_trace.set({})


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

    def __init__(self, message, refs=[]):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.traceback_refs = infer_trace.get()


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    from ..debug.label import label
    return MyiaTypeError(
        f"Wrong number of arguments for '{label(ident)}':"
        f" expected {expected}, got {got}."
    )


def check_nargs(ident, expected, args):
    """Return a MyiaTypeError for number of arguments mismatch."""
    got = len(args)
    if expected is not None and got != expected:
        raise type_error_nargs(ident, expected, got)


class TypeMismatchError(MyiaTypeError):
    """Error to generate when expecting a type and getting another."""

    def __init__(self, expected, got):
        """Initialize a TypeMismatchError."""
        message = f'Expected {expected}, but got {got}'
        super().__init__(message)
        self.expected = expected
        self.got = got


class MyiaInputTypeError(TypeError):
    """Represents a type error on an input to a Myia function."""
