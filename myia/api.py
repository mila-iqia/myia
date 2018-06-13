"""User-friendly interfaces to Myia machinery."""

import operator
from types import FunctionType
from typing import Any, List

from . import parser
from .cconv import closure_convert
from .infer import InferenceEngine
from .ir import Graph, clone
from .opt import PatternEquilibriumOptimizer, lib as optlib
from .pipeline import PipelineStep, PipelineDefinition
from .prim import py_implementations, vm_implementations, ops as P
from .prim.value_inferrers import ValueTrack, value_inferrer_constructors
from .prim.type_inferrers import TypeTrack, type_inferrer_constructors
from .prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from .specialize import TypeSpecializer
from .utils import TypeMap
from .vm import VM


default_object_map = {
    operator.add: P.add,
    operator.sub: P.sub,
    operator.mul: P.mul,
    operator.truediv: P.div,
    operator.mod: P.mod,
    operator.pow: P.pow,
    operator.eq: P.eq,
    operator.ne: P.ne,
    operator.lt: P.lt,
    operator.gt: P.gt,
    operator.le: P.le,
    operator.ge: P.ge,
    operator.pos: P.uadd,
    operator.neg: P.usub,
    operator.not_: P.not_,
    operator.getitem: P.getitem,
    operator.setitem: P.setitem,
    getattr: P.getattr,
    setattr: P.setattr,
}


def _convert_identity(env, x):
    return x


def _convert_sequence(env, seq):
    return type(seq)(env(x) for x in seq)


def _convert_function(env, fn):
    g = clone(parser.parse(fn))
    env.object_map[fn] = g
    return g


lax_type_map = TypeMap({
    FunctionType: _convert_function,
    tuple: _convert_sequence,
    list: _convert_sequence,
    object: _convert_identity,
    type: _convert_identity,
})


def parse(func: FunctionType, resolve_globals=True) -> Graph:
    """Parse a function into ANF."""
    pdef = standard_pipeline.select('parse', 'resolve')
    pdef = pdef.configure(resolve=resolve_globals)
    g = pdef.make()(input=func)['graph']
    return g


def run(g: Graph, args: List[Any]) -> Any:
    """Evaluate a graph on a set of arguments."""
    pdef = standard_pipeline.select('export')
    f = pdef.make()(graph=g)['output']
    return f(*args)


def compile(obj):
    """Return a version of the function that runs using Myia's VM."""
    pdef = standard_pipeline.select(
        'parse', 'resolve', 'export'
    )
    return pdef.make()(input=obj)['output']


############
# Pipeline #
############


class Converter(PipelineStep):
    """Convert a Python object into an object that can be in a Myia graph."""

    def __init__(self, pipeline_init, object_map, converters):
        """Initialize a Converter."""
        super().__init__(pipeline_init)
        self.object_map = dict(object_map)
        for prim, impl in self.resources.py_implementations.items():
            self.object_map[impl] = prim
        self.converters = converters

    def __call__(self, value):
        """Convert a value."""
        try:
            return self.object_map[value]
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

    def __init__(self, pipeline_init, tracks, required_tracks, timeout):
        """Initialize an Inferrer."""
        super().__init__(pipeline_init)
        self.tracks = tracks
        self.required_tracks = required_tracks
        self.timeout = timeout

    def step(self, graph, argspec):
        """Infer types, shapes, values, etc. for the graph."""
        argprops = argspec
        engine = InferenceEngine(
            self.pipeline,
            graph, argprops,
            tracks=self.tracks,
            required_tracks=self.required_tracks,
            timeout=self.timeout
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


class Exporter(PipelineStep):
    """Pipeline step to export a callable.

    Inputs:
        graph: The graph to wrap into a callable.

    Outputs:
        output: The callable.
    """

    def __init__(self, pipeline_init, implementations):
        """Initialize an Exporter."""
        super().__init__(pipeline_init)
        self.vm = VM(self.pipeline, implementations)

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
            constructors=value_inferrer_constructors
        ),
        type=TypeTrack.partial(
            constructors=type_inferrer_constructors
        ),
        shape=ShapeTrack.partial(
            constructors=shape_inferrer_constructors
        )
    ),
    required_tracks=['type'],
    timeout=1
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


step_export = Exporter.partial(
    implementations=vm_implementations
)


standard_pipeline = PipelineDefinition(
    resources=dict(
        py_implementations=py_implementations,
        convert=Converter.partial(
            object_map=default_object_map,
            converters=lax_type_map
        )
    ),
    steps=dict(
        parse=step_parse,
        resolve=step_resolve,
        infer=step_infer,
        specialize=step_specialize,
        opt=step_opt,
        cconv=step_cconv,
        export=step_export,
    )
)
