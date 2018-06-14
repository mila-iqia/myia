"""Tools to generate and configure Myia's operation pipeline."""


import inspect
from .utils import TypeMap, Named


# Use in a merge to indicate that a key should be deleted
DELETE = Named('DELETE')


# Placeholder for MergeMode.__init__
_ABSENT = Named('_ABSENT')


class MergeMode:
    """Wraps a value to control how it is merged.

    Class attributes:
        mode: The merge mode to use.

    Attributes:
        value: The value to merge.

    """

    mode = 'merge'  # NOTE: This is the default merge mode used by merge()

    def __init__(self, __value=_ABSENT, **kwargs):
        """Initialize a MergeMode.

        The value to merge will be the single argument given, or if there is no
        positional argument, the keywords dictionary.
        """
        if __value is _ABSENT:
            self.value = kwargs
        else:
            assert not kwargs
            self.value = __value


class Merge(MergeMode):
    """Merge normally."""

    mode = 'merge'


class Override(MergeMode):
    """Do not concatenate sequences."""

    mode = 'override'


class Reset(MergeMode):
    """Throw away the previous value."""

    mode = 'reset'


_cleanup_map = TypeMap()


@_cleanup_map.register(object)
def _cleanup_object(value):
    return value


@_cleanup_map.register(MergeMode)
def _cleanup_MergeMode(mm):
    return mm.value


@_cleanup_map.register(dict)
def _cleanup_dict(d):
    return type(d)({k: cleanup(v) for k, v in d.items() if v is not DELETE})


def _cleanup_sequence(xs):
    return type(xs)(cleanup(x) for x in xs)


_cleanup_map.register(tuple, _cleanup_sequence)
_cleanup_map.register(list, _cleanup_sequence)
_cleanup_map.register(set, _cleanup_sequence)


def cleanup(x):
    """Remove all MergeMode and DELETE instances from the data."""
    return _cleanup_map[type(x)](x)


_merge_map = TypeMap(discover=lambda cls: getattr(cls, '__merge__', None))


@_merge_map.register(dict)
def _merge_dict(d1, d2, mode):
    if mode == 'reset':
        return type(d1)(d2)

    rval = type(d1)()
    for k, v in d1.items():
        if k in d2:
            v2 = d2[k]
            if v2 is DELETE:
                pass
            else:
                rval[k] = merge(v, v2, mode)
        else:
            rval[k] = v
    for k, v in d2.items():
        if k not in d1:
            rval[k] = cleanup(v)
    return rval


@_merge_map.register(tuple)
def _merge_tuple(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@_merge_map.register(list)
def _merge_list(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@_merge_map.register(set)
def _merge_set(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs | ys
    else:
        return ys


@_merge_map.register(object)
def _merge_object(a, b, mode):
    return cleanup(b)


def merge(a, b, mode=MergeMode.mode):
    """Merge two data structures.

    Arguments:
        a: The original data.
        b: The data to merge.
        mode:
            'merge': (default) Sequences will be concatenated, sets merged,
                and dictionaries merged according to their keys.
            'override': Dictionaries are merged, but sequences are not
                concatenated.
            'reset': b is returned, or takes primacy over the data in a.
    """
    if isinstance(b, MergeMode):
        mode = b.mode
        b = b.value
    assert not isinstance(a, MergeMode)
    return _merge_map[type(a)](a, b, mode)


def _filter_keywords(f, kw):
    spec = inspect.getfullargspec(f)
    if spec.varkw:
        return kw, {}
    valid = spec.args + spec.kwonlyargs

    good = {k: v for k, v in kw.items() if k in valid}
    bad = {k: v for k, v in kw.items() if k not in valid}

    return good, bad


class NS:
    """Namespace for pipeline stages.

    This namespace preserves key order in the representation, unlike
    types.SimpleNamespace.
    """

    def __init__(self, **kwargs):
        """Initialize NS."""
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __repr__(self):
        args = [f'{k}={v}' for k, v in self.__dict__.items()]
        return f'NS({", ".join(args)})'


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
        _, invalid = _filter_keywords(f, self.keywords)
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


class PipelineDefinition:
    """Defines a Pipeline.

    A PipelineDefinition can be configured in a fine-grained manner.
    Call `make()` to instantiate a Pipeline that can be executed.

    Attributes:
        resources: An ensemble of common resources, pooled by the whole
            pipeline.
        steps: A sequence of steps. Each step should be a Partial on a
            PipelineStep.

    """

    def __init__(self, *, resources=None, steps):
        """Initialize a PipelineDefinition."""
        self.resources = resources or {}
        self.steps = steps
        self.step_names = list(steps.keys())

    def index(self, step, upper_bound=False):
        """Return the index corresponding to the given step.

        Args:
            step: The name of the step, or an int (returned without
                modification).
            upper_bound: Whether this is meant to be used as an
                upper bound or not. If it is True, then:
                * 1 will be added to the result if the step is a
                  string, so that it becomes inclusive,
                * unless it starts with !
                * Integers are still returned without modification.
        """
        if isinstance(step, str):
            if step.startswith('!'):
                return self.step_names.index(step[1:])
            else:
                plus = 1 if upper_bound else 0
                return self.step_names.index(step) + plus
        elif isinstance(step, int) or step is None:
            return step
        else:
            raise TypeError('index should be a string or int')

    def getslice(self, item):
        """Transform an item or a string slice into a numeric slice.

        The index for a string corresponds to a step name.

        >>> pd = PipelineDefinition(steps=dict(a=..., b=..., c=...))
        >>> pd.getslice("b")
        slice(1, 2)
        >>> pd.getslice(slice("b", "c"))
        slice(1, 3)
        """
        if isinstance(item, slice):
            assert item.step is None
            return slice(self.index(item.start),
                         self.index(item.stop, True))
        else:
            return slice(0, self.index(item, True))

    def insert_before(self, step=0, **new_steps):
        """Insert new steps before the given step.

        If no step is given, the new steps are inserted at the very beginning.

        This returns a new PipelineDefinition.
        """
        steps = list(self.steps.items())
        index = self.index(step) or 0
        steps[index:index] = new_steps.items()
        return PipelineDefinition(
            resources=self.resources,
            steps=dict(steps)
        )

    def insert_after(self, step=None, **new_steps):
        """Insert new steps after the given step.

        If no step is given, the new steps are inserted at the very end.

        This returns a new PipelineDefinition.
        """
        steps = list(self.steps.items())
        if step is None:
            steps.extend(new_steps.items())
        else:
            index = self.index(step)
            steps[index + 1:index + 1] = new_steps.items()
        return PipelineDefinition(
            resources=self.resources,
            steps=dict(steps)
        )

    def configure_resources(self, **resources):
        """Change resources or define new resources.

        This returns a new PipelineDefinition.
        """
        return PipelineDefinition(
            resources=merge(self.resources, resources, mode='override'),
            steps=self.steps
        )

    def configure(self, changes={}, **kwchanges):
        """Configure existing resources and steps.

        To add new steps, use `insert_before` or `insert_after`. To add new
        resources, use `configure_resources`.

        This returns a new PipelineDefinition.

        By default, new data is merged. These are all equivalent:

        >>> pd.configure(mystep={'param': 234})
        >>> pd.configure(mystep=Merge(param=234))
        >>> pd.configure({'mystep.param': 234})

        `mystep` will keep all of its previous parameters, except for the
        overriden one. If the parameter is a list, the list given in the
        update will be appended to the previous, but you can use
        `Override([a, b, c])` to replace the old list.
        """
        new_data = {**self.steps, **self.resources}
        changes = {**changes, **kwchanges}
        for path, delta in changes.items():
            if isinstance(delta, bool) and path in self.step_names:
                delta = Merge(pipeline_init={'active': delta})
            top, *parts = path.split('.')
            for part in reversed(parts):
                delta = {part: delta}
            new_data[top] = merge(new_data[top], delta, mode='override')
        return PipelineDefinition(
            resources={k: new_data[k] for k in self.resources},
            steps={k: new_data[k] for k in self.step_names}
        )

    def select(self, *names):
        """Define a pipeline with the listed steps.

        The names don't have to be in the same order as the original
        definition, so pipeline steps can be reordered this way.
        """
        return PipelineDefinition(
            resources=self.resources,
            steps={name: self.steps[name] for name in names}
        )

    def make(self):
        """Create a Pipeline from this definition."""
        return Pipeline(self)

    def run(self, **args):
        """Run a Pipeline made from this definition."""
        return self.make()(**args)

    def __getitem__(self, item):
        """Return a pipeline that only contains a subset of the steps."""
        steps = list(self.steps.items())
        return PipelineDefinition(
            resources=self.resources,
            steps=dict(steps[self.getslice(item)])
        )


class Pipeline:
    """Sequence of connected processing steps.

    Created from a PipelineDefinition. Each step processes the output of the
    previous step. A Pipeline also defines common resources for all the steps.
    """

    def __init__(self, defn):
        """Initialize a Pipeline from a PipelineDefinition."""
        def convert(x, name):
            if isinstance(x, Partial):
                x = x.partial(pipeline_init={'pipeline': self, 'name': name})
                x = x()
            return x

        self.defn = defn
        self.resources = NS()
        self.steps = NS()
        self._seq = []
        for k, v in defn.resources.items():
            self.resources[k] = convert(v, None)
        for k, v in defn.steps.items():
            v = convert(v, k)
            self.steps[k] = v
            self._seq.append(v)

    def __getitem__(self, item):
        """Return a sub-pipeline.

        item may be the name of a step, in which case the resulting
        sub-pipeline goes from the beginning to that step (inclusive), or it
        may be a range that starts at a given step and ends at another
        (again, inclusive).
        """
        return _PipelineSlice(self, item)

    def __call__(self, **args):
        """Execute the pipeline from start to finish."""
        return self[:](**args)


class _PipelineSlice:
    def __init__(self, pipeline, slice):
        self.pipeline = pipeline
        self.slice = self.pipeline.defn.getslice(slice)

    def __call__(self, **args):
        for step in self.pipeline._seq[self.slice]:
            if step.active:
                valid_args, rest = _filter_keywords(step.step, args)
                args = {**args, **step.step(**valid_args)}
        return args


class PipelineResource(Partializable):
    """Resource in a Pipeline."""

    def __init__(self, pipeline_init):
        """Initialize a PipelineStep."""
        self.active = pipeline_init.get('active', True)
        self.name = pipeline_init['name']
        self.pipeline = pipeline_init['pipeline']
        self.steps = self.pipeline.steps
        self.resources = self.pipeline.resources


class PipelineStep(PipelineResource):
    """Step in a Pipeline."""

    def step(self, **kwargs):
        """Execute this step only.

        The received arguments come from the previous step.
        """
        raise NotImplementedError()

    def __call__(self, **args):
        """Execute the pipeline all the way up to this step."""
        return self.pipeline[:self.name](**args)
