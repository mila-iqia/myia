"""Pre-made pipelines."""

from ..abstract import abstract_inferrer_constructors, Context
from ..compile import load_backend
from ..ir import GraphManager
from ..prim import py_registry
from ..pipeline.resources import scalar_object_map, standard_object_map, \
    standard_method_map, default_convert, ConverterResource, \
    InferenceResource

from . import steps
from .pipeline import PipelineDefinition


standard_resources = dict(
    manager=GraphManager.partial(),
    py_implementations=py_registry,
    method_map=standard_method_map,
    convert=ConverterResource.partial(
        object_map=standard_object_map,
        converter=default_convert
    ),
    inferrer=InferenceResource.partial(
        constructors=abstract_inferrer_constructors,
        context_class=Context,
    ),
    backend=None
)


######################
# Pre-made pipelines #
######################


_standard_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        erase_class=steps.step_erase_class,
        opt=steps.step_opt,
        opt2=steps.step_opt2,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        compile=steps.step_compile,
        wrap=steps.step_wrap,
    )
)

_backend = load_backend(None)
standard_pipeline = _backend.configure(_standard_pipeline)
del _backend

scalar_pipeline = standard_pipeline.configure({
    'convert.object_map': scalar_object_map,
})


standard_debug_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        erase_class=steps.step_erase_class,
        opt=steps.step_opt,
        opt2=steps.step_opt2,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        export=steps.step_debug_export,
        wrap=steps.step_wrap,
    )
).configure(backend=load_backend('numpy'))


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
