"""Misc utilities for inference."""


from contextvars import ContextVar

from ..dtype import Problem
from ..utils import Named, Event, Partializable, eprint


infer_trace = ContextVar('infer_trace')
infer_trace.set({})


# Represents an unknown value
ANYTHING = Named('ANYTHING')

# Represents inference problems
VOID = Named('VOID')

# Represents specialization problems
DEAD = Named('DEAD')
POLY = Named('POLY')
INACCESSIBLE = Named('INACCESSIBLE')


class Unspecializable(Exception):
    """Raised when it is impossible to specialize an inferrer."""

    def __init__(self, problem):
        """Initialize Unspecializable."""
        problem = Problem[problem]
        super().__init__(problem)
        self.problem = problem


class InferenceError(Exception, Partializable):
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
        self.traceback_refs = infer_trace.get()
        if app is not None:
            self.traceback_refs[app.context] = app

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        eprint(f'{type(self).__qualname__}: {self.message}')


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        if fn_ctx is None:
            super().print_tb_end(fn_ctx, args_ctx, is_prim)
            return
        s = f'{type(self).__qualname__}: `{fn_ctx}` cannot be called with' \
            f' argument types {args_ctx}.'
        if is_prim:
            s += f' Reason: {self.message}'
        eprint(s)


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""
