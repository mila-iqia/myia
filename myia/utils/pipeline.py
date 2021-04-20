"""Tools to generate and configure Myia's operation pipeline."""

import inspect
from types import FunctionType


def partition_keywords(f, kw):
    """Partitions keywords into compatible and incompatible with f.

    Returns:
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


class Pipeline:
    """Represents a sequence of function applications.

    ``Pipeline(f, g, h, arguments=args)(x=1, y=2)`` is roughly equivalent to
    ``h(**g(**f(x=1, y=2, **args)))``.

    Each function in the sequence only receives the arguments it can accept.

    Arguments:
        steps: A sequence of functions to call.
        arguments: Default arguments for the first invocation.
        name: The name of the Pipeline.
    """

    def __init__(self, *steps, arguments={}, name="pipeline"):
        """Initialize a Pipeline."""
        self.steps = steps
        self.arguments = arguments
        self.name = name

    def with_steps(self, *steps):
        """Return a new Pipeline using the given sequence of steps."""
        return type(self)(*steps, arguments=self.arguments, name=self.name)

    def without_step(self, step):
        """Return a new Pipeline without the given step."""
        idx = self.steps.index(step)
        return self.with_steps(*self[:idx], *self[idx + 1 :])

    def insert_after(self, base_step, *more_steps):
        """Insert new steps after the given step.

        This returns a new Pipeline.
        """
        idx = self.steps.index(base_step) + 1
        return self.with_steps(*self[:idx], *more_steps, *self[idx:])

    def make_transformer(self, in_key, out_key):
        """Create a callable for specific input and output keys.

        Arguments:
            in_key: The name of the pipeline input to use for the
                callable's argument.
            out_key: The name of the pipeline output to return.
        """

        def run(arg):
            res = self(**{in_key: arg})
            return res[out_key]

        return run

    def _call(self, fn, kwargs):
        """Execute one of the steps on the kwargs."""
        if not isinstance(fn, FunctionType):
            fn = fn.__call__
        valid_args, rest = partition_keywords(fn, kwargs)
        results = fn(**valid_args)
        if not isinstance(results, dict) and len(valid_args) == 1:
            (field_name,) = valid_args.keys()
            results = {field_name: results}
        return {**kwargs, **results}

    def __call__(self, **kwargs):
        """Execute the function sequence."""
        kwargs = {**self.arguments, **kwargs}
        for idx, fn in enumerate(self):
            kwargs = self._call(fn, kwargs)
        return kwargs

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, x):
        return self.steps[x]


class LoopPipeline(Pipeline):
    """Pipeline that loops over its steps while the "changes" argument is True.

    Arguments:
        steps: A sequence of functions to call.
        arguments: Default arguments for the first invocation.
        name: The name of the Pipeline.
        changes_field: The name of the field that tracks whether there were
            changes or not. Defaults to "changes".
    """

    def __init__(
        self, *steps, arguments={}, name="pipeline", changes_field="changes"
    ):
        """Initialize a LoopPipeline."""
        super().__init__(*steps, arguments=arguments, name=name)
        self.changes_field = changes_field

    def __call__(self, **kwargs):
        """Execute the function sequence."""
        kwargs = {**self.arguments, **kwargs}
        changes = True
        while changes:
            changes = kwargs[self.changes_field] = False
            for idx, fn in enumerate(self):
                kwargs = self._call(fn, kwargs)
                changes = changes or kwargs.get(self.changes_field, False)
        return kwargs
