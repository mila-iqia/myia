"""Tools to generate and configure Myia's operation pipeline."""

from time import perf_counter
from ..utils import merge, Merge, NS, Partial, Partializable, \
    partition_keywords


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

    def make_transformer(self, in_key, out_key):
        """Create a callable for specific input and output keys.

        Arguments:
            in_key: The name of the pipeline input to use for the
                callable's argument.
            out_key: The name of the pipeline output to return.
        """
        def run(arg):
            res = self.run(**{in_key: arg})
            return res[out_key]
        return run

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
                try:
                    x = x.partial(
                        pipeline_init={'pipeline': self, 'name': name}
                    )
                except TypeError:
                    pass
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

    def run_and_catch(self, **args):
        """Run the pipeline and catch errors, if any.

        Errors are put in the 'error' key of the result, and the step
        at which an error happened is put in the 'error_step' key.
        """
        profile = args.get('profile', True)
        if profile:
            profd = dict()
            gstart = perf_counter()
        for step in self.pipeline._seq[self.slice]:
            if 'error' in args:
                break
            if step.active:
                valid_args, rest = partition_keywords(step.step, args)
                try:
                    if profile:
                        start = perf_counter()
                    results = step.step(**valid_args)
                    if profile:
                        end = perf_counter()
                    if not isinstance(results, dict) and len(valid_args) == 1:
                        field_name, = valid_args.keys()
                        results = {field_name: results}
                    if profile:
                        subp = results.pop('profile', end - start)
                        if isinstance(subp, dict):
                            subp['__total__'] = end - start
                        profd[step.name] = subp
                    args = {**args, **results}
                except Exception as e:
                    args['error'] = e
                    args['error_step'] = step
        if profile:
            gend = perf_counter()
            profd['__total__'] = gend - gstart
            args['profile'] = profd
        return args

    def __call__(self, **args):
        results = self.run_and_catch(**args)
        if 'error' in results:
            raise results['error']
        return results


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


def pipeline_function(fn):
    """Create a pipeline step from a function.

    The provided function must receive `self` as its first argument, and
    then the pipeline fields it needs (the variable names are inspected
    by the pipeline to determine what values to give this step).
    """
    class PipelineStepFunction(PipelineStep):
        step = fn
    return PipelineStepFunction.partial()
