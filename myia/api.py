"""User-friendly interfaces to Myia machinery."""

import inspect
import numpy as np
from types import FunctionType

from . import dtype, parser, composite as C, operations
from .cconv import closure_convert
from .infer import InferenceEngine, ANYTHING
from .ir import Graph, clone, GraphManager
from .opt import PatternEquilibriumOptimizer, lib as optlib
from .pipeline import PipelineStep, PipelineResource, PipelineDefinition
from .prim import py_implementations, vm_implementations, ops as P
from .prim.value_inferrers import ValueTrack, value_inferrer_constructors
from .prim.type_inferrers import TypeTrack, type_inferrer_constructors
from .prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from .specialize import TypeSpecializer
from .utils import TypeMap, as_frozen
from .vm import VM
from .compile import step_wrap_primitives, step_compile, step_link, step_export


scalar_object_map = {
    operations.add: P.scalar_add,
    operations.sub: P.scalar_sub,
    operations.mul: P.scalar_mul,
    operations.truediv: P.scalar_div,
    operations.mod: P.scalar_mod,
    operations.pow: P.scalar_pow,
    operations.eq: P.scalar_eq,
    operations.ne: P.scalar_ne,
    operations.lt: P.scalar_lt,
    operations.gt: P.scalar_gt,
    operations.le: P.scalar_le,
    operations.ge: P.scalar_ge,
    operations.pos: P.scalar_uadd,
    operations.neg: P.scalar_usub,
    operations.not_: P.bool_not,
    operations.and_: P.bool_and,
    operations.or_: P.bool_or,
    operations.matmul: P.dot,
    operations.getitem: C.getitem,
    operations.setitem: C.setitem,
    operations.bool: P.identity,
    operations.getattr: P.getattr,
    operations.setattr: P.setattr,
    operations.len: C._len,
    operations.make_tuple: P.make_tuple,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.if_: P.if_,
}


standard_object_map = {
    operations.add: C.add,
    operations.sub: C.sub,
    operations.mul: C.mul,
    operations.truediv: C.div,
    operations.mod: C.mod,
    operations.pow: C.pow,
    operations.eq: C.eq,
    operations.ne: C.ne,
    operations.lt: C.lt,
    operations.gt: C.gt,
    operations.le: C.le,
    operations.ge: C.ge,
    operations.pos: C.uadd,
    operations.neg: C.usub,
    operations.not_: C.not_,
    operations.and_: C.and_,
    operations.or_: C.or_,
    operations.matmul: C.matmul,
    operations.getitem: C.getitem,
    operations.setitem: C.setitem,
    operations.bool: C.bool,
    operations.getattr: P.getattr,
    operations.setattr: P.setattr,
    operations.len: C._len,
    operations.make_tuple: P.make_tuple,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.if_: P.if_,
}


standard_method_map = TypeMap({
    dtype.Bool: {
        '__and__': P.bool_and,
        '__or__': P.bool_or,
        '__bool__': P.identity,
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
        '__bool__': C.int_bool,
        '__myia_to_array__': P.scalar_to_array,
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
        '__bool__': C.float_bool,
        '__myia_to_array__': P.scalar_to_array,
    },
    dtype.Tuple: {
        '__len__': P.tuple_len,
        '__getitem__': P.tuple_getitem,
        '__setitem__': P.tuple_setitem,
        '__myia_iter__': P.identity,
        '__myia_next__': C.tuple_next,
        '__myia_hasnext__': C.tuple_hasnext,
    },
    dtype.List: {
        '__len__': P.list_len,
        '__getitem__': P.list_getitem,
        '__setitem__': P.list_setitem,
        '__myia_iter__': C.list_iter,
    },
    dtype.Array: {
        '__add__': C.array_add,
        '__sub__': C.array_sub,
        '__mul__': C.array_mul,
        '__truediv__': C.array_div,
        '__mod__': C.array_mod,
        '__pow__': C.array_pow,
        '__pos__': C.array_uadd,
        '__neg__': C.array_usub,
        '__eq__': C.array_eq,
        '__ne__': C.array_ne,
        '__lt__': C.array_lt,
        '__gt__': C.array_gt,
        '__le__': C.array_le,
        '__ge__': C.array_ge,
        '__matmul__': P.dot,
        '__len__': P.array_len,
        '__getitem__': P.array_getitem,
        '__setitem__': P.array_setitem,
        '__myia_iter__': C.array_iter,
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
            tuple: dtype.Tuple,
            list: dtype.List,
        }
        mmap = self.resources.method_map
        for t1, t2 in type_map.items():
            for name, prim in mmap[t2].items():
                method = getattr(t1, name, None)
                if method is not None:
                    self.object_map[method] = _Unconverted(prim)

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

    def __init__(self, pipeline_init, pre, opts, post):
        """Initialize an Optimizer."""
        super().__init__(pipeline_init)
        self.pre = [opt(optimizer=self) for opt in pre]
        self.opts = opts
        self.post = [opt(optimizer=self) for opt in post]

    def step(self, graph):
        """Optimize the graph using the given patterns."""
        eq = PatternEquilibriumOptimizer(*self.opts, optimizer=self)
        seq = [*self.pre, eq, *self.post]
        for opt in seq:
            opt(graph)
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
        self.engine = InferenceEngine(
            self.pipeline,
            tracks=self.tracks,
            required_tracks=self.required_tracks,
        )

    def fill_in(self, argspec):
        """Fill in argspec with values for all tracks.

        The 'value' track will also be filled in even if it already exists,
        since it needs to be wrapped for the inferrer to use it.
        """
        for arg in argspec:
            if 'value' in arg:
                v = arg['value']
                for track_name, track in self.engine.tracks.items():
                    if track_name not in arg or track_name == 'value':
                        arg[track_name] = track.from_value(v, None)

    def step(self, graph, argspec):
        """Infer types, shapes, values, etc. for the graph."""
        engine = self.engine
        self.fill_in(argspec)
        try:
            res, context = engine.run(graph, argspec)
            return {'inference_results': res,
                    'inference_context': context,
                    'inferrer': engine}
        except Exception as exc:
            # We still want to keep the inferrer around even
            # if an error occurred.
            return {'error': exc,
                    'error_step': self,
                    'inferrer': engine}


class Specializer(PipelineStep):
    """Pipeline step to specialize a graph.

    Inputs:
        graph: The graph to specialize.
        inferrer: The inference engine.

    Outputs:
        graph: The specialized graph.
    """

    def step(self, graph, inferrer, inference_context):
        """Specialize the graph according to argument types."""
        spc = TypeSpecializer(inferrer)
        result = spc.run(graph, inference_context)
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
    pre=[],
    opts=[optlib.resolve_globals],
    post=[],
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
    pre=[],
    opts=[
        optlib.simplify_always_true,
        optlib.simplify_always_false,
        optlib.inline_unique_uses,
        optlib.simplify_partial,
        optlib.replace_applicator,
        optlib.elim_identity,
    ],
    post=[],
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
        wrap_primitives=step_wrap_primitives,
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
        inf = pip.steps.infer
        argspec = tuple({'value': arg} for arg in args)
        inf.fill_in(argspec)
        argnames = inspect.getargspec(self.fn).args
        for arg, name in zip(argspec, argnames):
            if name not in self.specialize_values:
                arg['value'] = ANYTHING
        key = as_frozen(argspec)
        if key not in self._cache:
            res = pip(
                input=self.fn,
                argspec=argspec
            )
            if 'error' in res:
                raise res['error']
            self._cache[key] = res
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
