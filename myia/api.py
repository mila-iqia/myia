"""User-friendly interfaces to Myia machinery."""

import operator
import numpy as np
from types import FunctionType

from . import dtype, parser
from .cconv import closure_convert
from .infer import InferenceEngine
from .ir import Graph, clone, GraphManager
from .opt import PatternEquilibriumOptimizer, lib as optlib
from .pipeline import PipelineStep, PipelineResource, PipelineDefinition
from .prim import py_implementations, vm_implementations, ops as P
from .prim.value_inferrers import ValueTrack, value_inferrer_constructors
from .prim.type_inferrers import TypeTrack, type_inferrer_constructors
from .prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from .specialize import TypeSpecializer
from .utils import TypeMap
from .vm import VM
from .compile import step_compile, step_link, step_export


scalar_object_map = {
    operator.add: P.scalar_add,
    operator.sub: P.scalar_sub,
    operator.mul: P.scalar_mul,
    operator.truediv: P.scalar_div,
    operator.mod: P.scalar_mod,
    operator.pow: P.scalar_pow,
    operator.eq: P.scalar_eq,
    operator.ne: P.scalar_ne,
    operator.lt: P.scalar_lt,
    operator.gt: P.scalar_gt,
    operator.le: P.scalar_le,
    operator.ge: P.scalar_ge,
    operator.pos: P.scalar_uadd,
    operator.neg: P.scalar_usub,
    operator.not_: P.bool_not,
    operator.and_: P.bool_and,
    operator.or_: P.bool_or,
    operator.getitem: P.getitem,
    operator.setitem: P.setitem,
    getattr: P.getattr,
    setattr: P.setattr,
}


# Provisional
standard_object_map = scalar_object_map


standard_method_map = TypeMap({
    dtype.Bool: {
        '__and__': P.bool_and,
        '__or__': P.bool_or,
    },
    dtype.Int: {
        '__add__': P.scalar_add,
        '__sub__': P.scalar_sub,
        '__mul__': P.scalar_mul,
        '__truediv__': P.scalar_div,
        '__mod__': P.scalar_mod,
        '__pow__': P.scalar_pow,
        '__pos__': P.scalar_uadd,
        '__neg__': P.scalar_usub,
        '__eq__': P.scalar_eq,
        '__ne__': P.scalar_ne,
        '__lt__': P.scalar_lt,
        '__gt__': P.scalar_gt,
        '__le__': P.scalar_le,
        '__ge__': P.scalar_ge,
    },
    dtype.Float: {
        '__add__': P.scalar_add,
        '__sub__': P.scalar_sub,
        '__mul__': P.scalar_mul,
        '__truediv__': P.scalar_div,
        '__mod__': P.scalar_mod,
        '__pow__': P.scalar_pow,
        '__pos__': P.scalar_uadd,
        '__neg__': P.scalar_usub,
        '__eq__': P.scalar_eq,
        '__ne__': P.scalar_ne,
        '__lt__': P.scalar_lt,
        '__gt__': P.scalar_gt,
        '__le__': P.scalar_le,
        '__ge__': P.scalar_ge,
    },
    dtype.Array: {
        # TODO
    }
})


def _convert_identity(env, x):
    return x


def _convert_sequence(env, seq):
    return type(seq)(env(x) for x in seq)


def _convert_function(env, fn):
    g = clone(parser.parse(fn))
    env.resources.manager.add_graph(g)
    env.object_map[fn] = g
    return g


lax_type_map = TypeMap({
    FunctionType: _convert_function,
    tuple: _convert_sequence,
    list: _convert_sequence,
    object: _convert_identity,
    type: _convert_identity,
})


############
# Pipeline #
############


class _Unconverted:
    # This is just used by Converter to delay conversion of graphs associated
    # to operators or methods until they are actually needed.
    def __init__(self, value):
        self.value = value


class Converter(PipelineResource):
    """Convert a Python object into an object that can be in a Myia graph."""

    def __init__(self, pipeline_init, object_map, converters):
        """Initialize a Converter."""
        super().__init__(pipeline_init)
        self.converters = converters
        self.object_map = {}
        for k, v in object_map.items():
            self.object_map[k] = _Unconverted(v)
        for prim, impl in self.resources.py_implementations.items():
            self.object_map[impl] = prim
        type_map = {
            bool: dtype.Bool,
            int: dtype.Int,
            float: dtype.Float,
            np.ndarray: dtype.Array,
            np.int8: dtype.Int,
            np.int16: dtype.Int,
            np.int32: dtype.Int,
            np.int64: dtype.Int,
            np.float16: dtype.Float,
            np.float32: dtype.Float,
            np.float64: dtype.Float,
        }
        mmap = self.resources.method_map
        for t1, t2 in type_map.items():
            for name, prim in mmap[t2].items():
                self.object_map[getattr(t1, name)] = _Unconverted(prim)

    def __call__(self, value):
        """Convert a value."""
        try:
            v = self.object_map[value]
            if isinstance(v, _Unconverted):
                v = v.value
                v = self.converters[type(v)](self, v)
                self.object_map[value] = v
            return v
        except (TypeError, KeyError):
            pass

        return self.converters[type(value)](self, value)


class Parser(PipelineStep):
    """Pipeline step to parse a function.

    Inputs:
        input: A function.

    Outputs:
        graph: A graph.
    """

    def step(self, input):
        """Assert that input is a Graph, and set it as the 'graph' key."""
        g = self.resources.convert(input)
        assert isinstance(g, Graph)
        return {'graph': g}


class Optimizer(PipelineStep):
    """Pipeline step to optimize a graph.

    Inputs:
        graph: The graph to optimize.

    Outputs:
        graph: The optimized graph.
    """

    def __init__(self, pipeline_init, opts):
        """Initialize an Optimizer."""
        super().__init__(pipeline_init)
        self.opts = opts

    def step(self, graph):
        """Optimize the graph using the given patterns."""
        eq = PatternEquilibriumOptimizer(*self.opts, optimizer=self)
        eq(graph)
        self.resources.manager.keep_roots(graph)
        return {'graph': graph}


class Inferrer(PipelineStep):
    """Pipeline step to run type/shape/value/etc. inference.

    Inputs:
        graph: The graph to infer.
        argspec: Information about argument types.

    Outputs:
        inference_results: Inference results for the graph's output.
        inferrer: The inference engine.
    """

    def __init__(self, pipeline_init, tracks, required_tracks):
        """Initialize an Inferrer."""
        super().__init__(pipeline_init)
        self.tracks = tracks
        self.required_tracks = required_tracks

    def step(self, graph, argspec):
        """Infer types, shapes, values, etc. for the graph."""
        argprops = argspec
        engine = InferenceEngine(
            self.pipeline,
            graph, argprops,
            tracks=self.tracks,
            required_tracks=self.required_tracks,
        )
        return {'inference_results': engine.output_info(),
                'inferrer': engine}


class Specializer(PipelineStep):
    """Pipeline step to specialize a graph.

    Inputs:
        graph: The graph to specialize.
        inferrer: The inference engine.

    Outputs:
        graph: The specialized graph.
    """

    def step(self, graph, inferrer):
        """Specialize the graph according to argument types."""
        spc = TypeSpecializer(inferrer)
        result = spc.result
        self.resources.manager.keep_roots(result)
        return {'graph': result}


class ClosureConverter(PipelineStep):
    """Pipeline step to closure convert a graph.

    Inputs:
        graph: The graph to closure convert.

    Outputs:
        graph: The closure converted graph.
    """

    def step(self, graph):
        """Closure convert the graph."""
        closure_convert(graph)
        return {'graph': graph}


class DebugVMExporter(PipelineStep):
    """Pipeline step to export a callable.

    Inputs:
        graph: The graph to wrap into a callable.

    Outputs:
        output: The callable.
    """

    def __init__(self, pipeline_init, implementations):
        """Initialize an DebugVMExporter."""
        super().__init__(pipeline_init)
        self.vm = VM(self.pipeline.resources.convert,
                     self.pipeline.resources.manager,
                     self.pipeline.resources.py_implementations,
                     implementations)

    def step(self, graph):
        """Make a Python callable out of the graph."""
        return {'output': self.vm.export(graph)}


step_parse = Parser.partial()


step_resolve = Optimizer.partial(
    opts=[optlib.resolve_globals]
)


step_infer = Inferrer.partial(
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
)


step_specialize = Specializer.partial()


step_opt = Optimizer.partial(
    opts=[
        optlib.simplify_always_true,
        optlib.simplify_always_false,
        optlib.inline_unique_uses,
    ]
)


step_cconv = ClosureConverter.partial()


step_debug_export = DebugVMExporter.partial(
    implementations=vm_implementations
)


_standard_pipeline = PipelineDefinition(
    resources=dict(
        manager=GraphManager.partial(),
        py_implementations=py_implementations,
        method_map=standard_method_map,
        convert=Converter.partial(
            object_map=standard_object_map,
            converters=lax_type_map
        ),
    ),
    steps=dict(
        parse=step_parse,
        resolve=step_resolve,
        infer=step_infer,
        specialize=step_specialize,
        opt=step_opt,
        cconv=step_cconv,
    )
)


######################
# Pre-made pipelines #
######################


standard_pipeline = _standard_pipeline \
    .insert_after(
        compile=step_compile,
        link=step_link,
        export=step_export
    )


standard_debug_pipeline = _standard_pipeline \
    .insert_after(export=step_debug_export)


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
