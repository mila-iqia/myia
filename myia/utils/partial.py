"""Utilities for partial application."""


import inspect


from .merge import merge


def partition_keywords(f, kw):
    """Partitions keywords into compatible and incompatible with f.

    Return:
        good: key/value pairs that the function f recognizes.
        bad: key/value pairs that the function f does not recognize.
    """
    spec = inspect.getfullargspec(f)
    if spec.varkw:
        return kw, {}
    valid = spec.args + spec.kwonlyargs

    good = {k: v for k, v in kw.items() if k in valid}
    bad = {k: v for k, v in kw.items() if k not in valid}

    return good, bad


class Partial:
    """Partial application of a function.

    This differs from functools.partial in a few ways:

    * Only keyword arguments are accepted.
    * Argument names are validated immediately.
    * It is possible to merge two partials, with the second updating the
      parameters of the first. It is also possible to merge a dict and
      a Partial. Merge and Override work basically like on dictionaries.
    * `merge(partial1, Override(partial2))` lets us change the
      constructor.
    """

    def __init__(self, func, **keywords):
        """Initialize a Partial."""
        self.func = func
        self.keywords = keywords
        self._validate()

    def _validate(self):
        """Check that all the argument names are valid."""
        if isinstance(self.func, type):
            f = getattr(self.func, '__init__', self.func)
        else:
            f = self.func
        _, invalid = partition_keywords(f, self.keywords)
        if invalid:
            keys = ", ".join(f"'{k}'" for k in invalid.keys())
            raise TypeError(f"{f} has no argument(s) named {keys}")

    def partial(self, **keywords):
        """Refine this Partial with additional keywords."""
        return merge(self, keywords)

    def __call__(self, **kwargs):
        """Merge stored arguments with kwargs and call the function."""
        return self.func(**self.keywords, **kwargs)

    def __merge__(self, partial, mode):
        """Combine arguments from two partials."""
        if isinstance(partial, dict):
            partial = Partial(self.func, **partial)

        assert isinstance(partial, Partial)

        if partial.func is self.func \
                or mode == 'override' or mode == 'reset':
            func = partial.func
        else:
            raise ValueError('Incompatible func')

        kwargs = merge(self.keywords, partial.keywords, mode)

        return Partial(func, **kwargs)

    def __repr__(self):
        args = [f'{k}={v}' for k, v in self.keywords.items()]
        return f'{self.func.__name__}({", ".join(args)})'


class Partializable:
    """Class for which partial instances may be created.

    This defines `Class.partial(arg=val, ...)`, which is equivalent to
    `Partial(Class, arg=val, ...)`.
    """

    @classmethod
    def partial(cls, **kwargs):
        """Return a Partial on this class constructor."""
        return Partial(cls, **kwargs)


class PartialCallable(Partializable):
    """TODO: Ask Arnaud."""

    def __new__(self, fn, pipeline_init, **kwargs):
        """TODO: Ask Arnaud."""
        return fn(**kwargs)
