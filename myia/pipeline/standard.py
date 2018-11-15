"""Pre-made pipelines."""


from ..compile import step_wrap_primitives, step_compile, step_link, \
    step_export
from ..infer import Context
from ..ir import GraphManager
from ..prim import py_implementations
from ..prim.value_inferrers import ValueTrack, value_inferrer_constructors
from ..prim.type_inferrers import TypeTrack, type_inferrer_constructors
from ..prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from ..pipeline.resources import scalar_object_map, standard_object_map, \
    standard_method_map, default_convert, ConverterResource, \
    InferenceResource

from . import steps
from .pipeline import PipelineDefinition


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
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        prepare=steps.step_prepare,
        opt=steps.step_opt,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        wrap_primitives=step_wrap_primitives,
        compile=step_compile,
        link=step_link,
        export=step_export,
        wrap=steps.step_wrap,
    )
)


standard_debug_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        prepare=steps.step_prepare,
        opt=steps.step_debug_opt,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        export=steps.step_debug_export,
        wrap=steps.step_wrap,
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
