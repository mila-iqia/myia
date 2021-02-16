"""Exceptions that may be raised within Myia."""

import warnings
from contextlib import contextmanager
from contextvars import ContextVar

infer_trace = ContextVar("infer_trace", default={})


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
        priority: Priority of this error, in the case that there are
            several.
    """

    def __init__(self, message, refs=[], pytb=None, priority=0):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.pytb = pytb
        self.priority = priority
        self.traceback_refs = infer_trace.get()


class InternalInferenceError(InferenceError):
    """This kind of error denotes a bug in Myia."""


class MyiaConversionError(InferenceError):
    """Error converting an object in a Myia program."""


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""


class MyiaValueError(InferenceError):
    """Value error in a Myia program."""


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


class MyiaNameError(InferenceError):
    """Raised when a name is not found in scope."""


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
    return args


class TypeMismatchError(MyiaTypeError):
    """Error to generate when expecting a type and getting another."""

    def __init__(self, expected, got):
        """Initialize a TypeMismatchError."""
        message = f"Expected {expected}, but got {got}"
        super().__init__(message)
        self.expected = expected
        self.got = got


class AnnotationMismatchError(MyiaTypeError):
    """Error when an annotation does not match an abstract."""


class MyiaInputTypeError(TypeError):
    """Represents a type error on an input to a Myia function."""


@contextmanager
def untested():
    """Wrap legacy code that isn't covered anymore but we are not sure why."""
    warnings.warn(
        UserWarning(
            "You triggered code that used to be essential but stopped being"
            " covered for some reason. Please report your use case so that we"
            " can validate the code again and add a test for it."
        ),
        stacklevel=3,
    )
    yield


__consolidate__ = True
__all__ = [
    "InferenceError",
    "InternalInferenceError",
    "MyiaAttributeError",
    "MyiaConversionError",
    "MyiaInputTypeError",
    "MyiaNameError",
    "MyiaShapeError",
    "MyiaTypeError",
    "MyiaValueError",
    "TypeMismatchError",
    "AnnotationMismatchError",
    "check_nargs",
    "infer_trace",
    "type_error_nargs",
    "untested",
]
