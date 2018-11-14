"""User-friendly interfaces to Myia machinery."""

import inspect

from .compile import step_wrap_primitives, step_compile, step_link, step_export
from .infer import Context, MyiaTypeError
from .ir import GraphManager
from .pipeline import PipelineDefinition
from .prim import py_implementations
from .prim.value_inferrers import ValueTrack, value_inferrer_constructors
from .prim.type_inferrers import TypeTrack, type_inferrer_constructors
from .prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from .resources import scalar_object_map, standard_object_map, \
    standard_method_map, default_convert, ConverterResource, \
    InferenceResource
from .steps import step_parse, step_resolve, step_infer, step_specialize, \
    step_prepare, step_debug_opt, step_opt, \
    step_validate, step_cconv, step_wrap, step_debug_export
from .utils import as_frozen


standard_resources = dict(
    manager=GraphManager.partial(),
    py_implementations=py_implementations,
    method_map=standard_method_map,
    convert=ConverterResource.partial(
        object_map=standard_object_map,
        converter=default_convert
    ),
    inferrer=InferenceResource.partial(
        tracks=dict(
            value=ValueTrack.partial(
                constructors=value_inferrer_constructors,
                max_depth=1
            ),
            type=TypeTrack.partial(
                constructors=type_inferrer_constructors
            ),
            shape=ShapeTrack.partial(
                constructors=shape_inferrer_constructors
            )
        ),
        required_tracks=['type'],
        tied_tracks={},
        context_class=Context,
        erase_value=True,
    )
)


######################
# Pre-made pipelines #
######################


standard_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=step_parse,
        resolve=step_resolve,
        infer=step_infer,
        specialize=step_specialize,
        prepare=step_prepare,
        opt=step_opt,
        cconv=step_cconv,
        validate=step_validate,
        wrap_primitives=step_wrap_primitives,
        compile=step_compile,
        link=step_link,
        export=step_export,
        wrap=step_wrap,
    )
)


standard_debug_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=step_parse,
        resolve=step_resolve,
        infer=step_infer,
        specialize=step_specialize,
        prepare=step_prepare,
        opt=step_debug_opt,
        cconv=step_cconv,
        validate=step_validate,
        export=step_debug_export,
        wrap=step_wrap,
    )
)


scalar_pipeline = standard_pipeline.configure({
    'convert.object_map': scalar_object_map
})


scalar_debug_pipeline = standard_debug_pipeline.configure({
    'convert.object_map': scalar_object_map
})


######################
# Pre-made utilities #
######################


scalar_parse = scalar_pipeline \
    .select('parse', 'resolve') \
    .make_transformer('input', 'graph')


scalar_debug_compile = scalar_debug_pipeline \
    .select('parse', 'resolve', 'export') \
    .make_transformer('input', 'output')


#################
# Top-level API #
#################


class MyiaFunction:
    """Represents a function compiled by Myia.

    MyiaFunction will compile the original function for every combination of
    argument types and shapes it is given (as well as their values,
    optionally).

    Attributes:
        fn: The root function to compile.
        specialize_values: Set of arguments for which we should specialize the
            function based on their values (list of argument names).

    """

    def __init__(self, fn, specialize_values=[]):
        """Initialize a MyiaFunction."""
        self.fn = fn
        self.specialize_values = set(specialize_values)
        self._cache = {}

    def specialize(self, args):
        """Specialize on the types of the given arguments.

        Returns a Pipeline. If the argument types were seen before, returns a
        cached version.
        """
        pip = standard_debug_pipeline.make()
        inf = pip.resources.inferrer

        argnames = inspect.getfullargspec(self.fn).args
        n1 = len(argnames)
        n2 = len(args)
        if n1 != n2:
            raise MyiaTypeError(
                f'Wrong number of arguments: expected {n1}, got {n2}'
            )

        argspec = tuple({'value': arg,
                         '_erase_value': name not in self.specialize_values}
                        for arg, name in zip(args, argnames))
        inf.fill_in(argspec)
        key = as_frozen(argspec)
        if key not in self._cache:
            self._cache[key] = pip(
                input=self.fn,
                argspec=argspec
            )
        return self._cache[key]

    def compile(self, args):
        """Returns a function specialized for the given args."""
        return self.specialize(args)['output']

    def __call__(self, *args):
        """Call the function on the given args."""
        return self.compile(args)(*args)


def myia(fn=None, *, specialize_values=[]):
    """Create a function using Myia's runtime.

    `@myia` can be used as a simple decorator. If custom options are needed,
    they can be provided as keyword arguments:

        @myia
        def myfun(x, y):
            return x + y

        @myia(specialize_values=['cond'])
        def myfun2(cond, x, y):
            return x if cond else y

    Arguments:
        fn: The Python function to convert.
        specialize_values: Set of arguments for which we should specialize the
            function based on their values (list of argument names).
    """
    if fn is None:
        def deco(fn):
            return MyiaFunction(fn, specialize_values)
        return deco
    else:
        return MyiaFunction(fn, specialize_values)
