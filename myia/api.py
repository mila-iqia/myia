"""User-friendly interfaces to Myia machinery."""

import inspect
import math
import numpy as np
from types import FunctionType

from . import dtype, parser, composite as C, operations
from .cconv import closure_convert
from .dtype import Tuple, List, Class, Array, Int, Float, Bool, \
    Number, Problem, tag_to_dataclass, ismyiatype, type_to_np_dtype, \
    TypeMeta
from .infer import InferenceEngine, Inferrer, ANYTHING, \
    Context, Contextless, CONTEXTLESS, reify
from .ir import Graph, clone, GraphManager
from .opt import PatternEquilibriumOptimizer, lib as optlib, CSE, \
    erase_class
from .pipeline import PipelineStep, PipelineResource, PipelineDefinition
from .prim import py_implementations, vm_implementations, ops as P
from .prim.value_inferrers import ValueTrack, value_inferrer_constructors
from .prim.type_inferrers import TypeTrack, type_inferrer_constructors
from .prim.shape_inferrers import ShapeTrack, shape_inferrer_constructors
from .specialize import TypeSpecializer
from .utils import TypeMap, as_frozen, overload, UNKNOWN, ErrorPool
from .vm import VM
from .compile import step_wrap_primitives, step_compile, step_link, step_export
from .validate import validate, whitelist as default_whitelist, ValidationError


scalar_object_map = {
    operations.add: P.scalar_add,
    operations.sub: P.scalar_sub,
    operations.mul: P.scalar_mul,
    operations.truediv: C.scalar_truediv,
    operations.floordiv: C.scalar_floordiv,
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
    operations.exp: P.scalar_exp,
    operations.log: P.scalar_log,
    operations.sin: P.scalar_sin,
    operations.cos: P.scalar_cos,
    operations.tan: P.scalar_tan,
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
    operations.make_list: P.make_list,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.to_array: C.to_array,
    operations.if_: P.if_,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: P.scalar_exp,
    math.log: P.scalar_log,
    math.sin: P.scalar_sin,
    math.cos: P.scalar_cos,
    math.tan: P.scalar_tan,
}


standard_object_map = {
    operations.add: C.add,
    operations.sub: C.sub,
    operations.mul: C.mul,
    operations.truediv: C.truediv,
    operations.floordiv: C.floordiv,
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
    operations.exp: C.exp,
    operations.log: C.log,
    operations.sin: C.sin,
    operations.cos: C.cos,
    operations.tan: C.tan,
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
    operations.make_list: P.make_list,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.to_array: C.to_array,
    operations.if_: P.if_,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: P.scalar_exp,
    math.log: P.scalar_log,
    math.sin: P.scalar_sin,
    math.cos: P.scalar_cos,
    math.tan: P.scalar_tan,
    np.floor: C.floor,
    np.trunc: C.trunc,
    np.add: C.add,
    np.subtract: C.sub,
    np.multiply: C.mul,
    np.divide: C.truediv,
    np.mod: C.mod,
    np.power: C.pow,
    np.exp: C.exp,
    np.log: C.log,
    np.sin: C.sin,
    np.cos: C.cos,
    np.tan: C.tan,
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
        '__floordiv__': C.int_floordiv,
        '__truediv__': C.int_truediv,
        '__mod__': P.scalar_mod,
        '__pow__': P.scalar_pow,
        '__floor__': P.identity,
        '__trunc__': P.identity,
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
    dtype.UInt: {
        '__add__': P.scalar_add,
        '__sub__': P.scalar_sub,
        '__mul__': P.scalar_mul,
        '__floordiv__': P.scalar_div,
        '__truediv__': C.int_truediv,
        '__mod__': P.scalar_mod,
        '__pow__': P.scalar_pow,
        '__floor__': P.identity,
        '__trunc__': P.identity,
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
        '__floordiv__': C.float_floordiv,
        '__truediv__': P.scalar_div,
        '__mod__': P.scalar_mod,
        '__pow__': P.scalar_pow,
        '__floor__': P.scalar_floor,
        '__trunc__': P.scalar_trunc,
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
        '__truediv__': C.array_truediv,
        '__floordiv__': C.array_floordiv,
        '__mod__': C.array_mod,
        '__pow__': C.array_pow,
        '__floor__': C.array_floor,
        '__trunc__': C.array_trunc,
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


@overload
def default_convert(env, fn: FunctionType):
    """Default converter for Python types."""
    g = clone(parser.parse(fn))
    env.resources.manager.add_graph(g)
    env.object_map[fn] = g
    return g


@overload  # noqa: F811
def default_convert(env, g: Graph):
    mng = env.resources.manager
    if g._manager is not mng:
        g2 = clone(g)
        env.object_map[g] = g2
        mng.add_graph(g2)
        return g2
    else:
        return g


@overload  # noqa: F811
def default_convert(env, seq: (tuple, list)):
    return type(seq)(env(x) for x in seq)


@overload  # noqa: F811
def default_convert(env, x: (object, type, TypeMeta)):
    return x


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

    def __init__(self, pipeline_init, object_map, converter):
        """Initialize a Converter."""
        super().__init__(pipeline_init)
        self.converter = converter
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
                v = self.converter(self, v.value)
                self.object_map[value] = v
            return v
        except (TypeError, KeyError):
            pass

        return self.converter(self, value)


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

    def __init__(self, pipeline_init, phases, run_only_once=False):
        """Initialize an Optimizer."""
        super().__init__(pipeline_init)
        self.run_only_once = run_only_once
        self.phases = []
        for name, spec in phases.items():
            if isinstance(spec, list):
                spec = PatternEquilibriumOptimizer(*spec, optimizer=self)
            else:
                spec = spec(optimizer=self)
            self.phases.append(spec)

    def step(self, graph):
        """Optimize the graph using the given patterns."""
        changes = True
        while changes:
            changes = False
            for opt in self.phases:
                if opt(graph):
                    changes = True
            if self.run_only_once:
                break
        self.resources.manager.keep_roots(graph)
        return {'graph': graph}


class InferrerStep(PipelineStep):
    """Pipeline step to run type/shape/value/etc. inference.

    Inputs:
        graph: The graph to infer.
        argspec: Information about argument types.

    Outputs:
        inference_results: Inference results for the graph's output.
        inferrer: The inference engine.
    """

    def __init__(self,
                 pipeline_init,
                 tracks,
                 required_tracks,
                 tied_tracks,
                 context_class):
        """Initialize an InferrerStep."""
        super().__init__(pipeline_init)
        self.tracks = tracks
        self.required_tracks = required_tracks
        self.tied_tracks = tied_tracks
        self.context_class = context_class
        self.engine = InferenceEngine(
            self.pipeline,
            tracks=self.tracks,
            tied_tracks=self.tied_tracks,
            context_class=self.context_class,
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
                    if track_name not in arg:
                        arg[track_name] = track.from_value(v, None)
                    else:
                        arg[track_name] = track.from_external(arg[track_name])
            else:
                for track_name, track in self.engine.tracks.items():
                    if track_name in arg:
                        arg[track_name] = track.from_external(arg[track_name])

    def step(self, graph, argspec):
        """Infer types, shapes, values, etc. for the graph."""
        engine = self.engine
        self.fill_in(argspec)
        try:
            res, context = engine.run(graph, argspec, self.required_tracks)
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


class _InferenceUpdater:

    def __init__(self, manager, inferrer):
        self.manager = manager
        self.inferrer = inferrer
        self.todo = set()
        manager.events.add_node.register(self._on_add_node)
        manager.events.drop_node.register(self._on_drop_node)
        manager.events.add_edge.register(self._on_add_edge)

    async def _run(self):
        todo, self.todo = self.todo, set()
        for node in todo:
            await self._update_type(node)

    def run(self):
        self.inferrer.run_coroutine(self._run())

    async def _update_type(self, node):
        ref = self.inferrer.ref(node, CONTEXTLESS)
        inferred = {}
        for track_name, track in self.inferrer.tracks.items():
            try:
                previous = await self.inferrer.invalidate(track_name, ref)
            except KeyError:
                previous = UNKNOWN
            result = await ref[track_name]
            inferred[track_name] = result
            expected = node.expect_inferred[track_name]
            if expected is not UNKNOWN:
                expected = await reify(track.from_external(expected))
                self.inferrer.equiv.declare_equivalent(result, expected, [ref])
            if previous is not UNKNOWN:
                previous = await reify(previous)
                self.inferrer.equiv.declare_equivalent(result, previous, [ref])
        node.inferred.update(inferred)

    def _on_add_node(self, event, node):
        self.todo.add(node)

    def _on_drop_node(self, event, node):
        if node in self.todo:
            self.todo.remove(node)

    def _on_add_edge(self, event, src, key, dest):
        src.expect_inferred.update(src.inferred)
        src.inferred.clear()
        self.todo.add(src)


class Preparator(PipelineStep):
    """Pipeline step to prepare the graph to optimization.

    This should be run on the specialized graph.

    Inputs:
        graph: The graph to prepare.

    Outputs:
        graph: The prepared graph.
    """

    def __init__(self, pipeline_init, erase_classes=True, watch=True):
        """Initialize a Preparator."""
        super().__init__(pipeline_init)
        self.erase_classes = erase_classes
        self.watch = watch
        self.inferrer = self.pipeline.resources.live_infer.engine

    def step(self, graph):
        """Prepare the graph."""
        mng = self.resources.manager
        if self.watch:
            upd = _InferenceUpdater(mng, self.inferrer)
            self.resources.inference_updater = upd
        if self.erase_classes:
            erase_class(graph, mng)
        if self.watch:
            self.resources.inference_updater.run()
        return {'graph': graph}


@reify.variant
async def _eliminate_inferrers(self, t: Inferrer):
    t2 = await t.as_function_type()
    if ismyiatype(t2, Problem):  # pragma: no cover
        raise ValidationError(f'{t} became {t2}')
    return t2


class Validator(PipelineStep):
    """Pipeline step to validate a graph prior to compilation.

    Inputs:
        graph: The graph to validate.

    Outputs:
        None.
    """

    def __init__(self, pipeline_init, whitelist=default_whitelist):
        """Initialize a Validator."""
        super().__init__(pipeline_init)
        self.whitelist = whitelist

    async def _eliminate_inferrers(self):
        errs = ErrorPool()
        for node in self.pipeline.resources.manager.all_nodes:
            try:
                node.type = await _eliminate_inferrers(node.type)
            except ValidationError as e:  # pragma: no cover
                exc = ValidationError(f'In {node}::{node.type}, {e.args[0]}')
                errs.add(exc)
        errs.trigger()

    def step(self, graph):
        """Validate the graph."""
        if hasattr(self.resources, 'inference_updater'):
            self.resources.inference_updater.run()
        self.pipeline.resources.live_infer.engine.run_coroutine(
            self._eliminate_inferrers()
        )
        validate(graph, whitelist=self.whitelist)
        return {}


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


@overload
def _convert_arg(arg, orig_t: Tuple, vm_t):
    if not isinstance(arg, tuple):
        raise TypeError('Expected tuple')
    oe = orig_t.elements
    ve = vm_t.elements
    if len(arg) != len(ve):
        raise TypeError(f'Expected {len(ve)} elements')
    return tuple(convert_arg(x, o, v)
                 for x, o, v in zip(arg, oe, ve))


@overload  # noqa: F811
def _convert_arg(arg, orig_t: List, vm_t):
    if not isinstance(arg, list):
        raise TypeError('Expected list')
    ot = orig_t.element_type
    vt = vm_t.element_type
    return [convert_arg(x, ot, vt) for x in arg]


@overload  # noqa: F811
def _convert_arg(arg, orig_t: Class, vm_t):
    # If the EraseClass opt was applied, vm_t may be Tuple
    dc = tag_to_dataclass[orig_t.tag]
    if not isinstance(arg, dc):
        raise TypeError(f'Expected {dc.__qualname__}')
    arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
    oe = list(orig_t.attributes.values())
    vm_is_tup = ismyiatype(vm_t, Tuple)
    if vm_is_tup:
        ve = vm_t.elements
    else:
        ve = vm_t.attributes.values()
    tup = tuple(convert_arg(x, o, v)
                for x, o, v in zip(arg, oe, ve))
    if vm_is_tup:
        return tup
    else:
        return dc(*tup)


@overload  # noqa: F811
def _convert_arg(arg, orig_t: Array, vm_t):
    if not isinstance(arg, np.ndarray):
        raise TypeError('Expected ndarray')
    et = orig_t.elements
    assert ismyiatype(et, Number)
    dtype = type_to_np_dtype(et)
    if arg.dtype != dtype:
        raise TypeError('Wrong dtype')
    return arg


@overload  # noqa: F811
def _convert_arg(arg, orig_t: Int, vm_t):
    if not isinstance(arg, int):
        raise TypeError(f'Expected int')
    return arg


@overload  # noqa: F811
def _convert_arg(arg, orig_t: Float, vm_t):
    if not isinstance(arg, float):
        raise TypeError(f'Expected float')
    return arg


@overload  # noqa: F811
def _convert_arg(arg, orig_t: Bool, vm_t):
    if not isinstance(arg, bool):
        raise TypeError(f'Expected bool')
    return arg


def convert_arg(arg, orig_t, vm_t):
    """Check that arg matches orig_t, and convert to vm_t."""
    return _convert_arg[orig_t](arg, orig_t, vm_t)


@overload
def _convert_result(res, orig_t, vm_t: Class):
    dc = tag_to_dataclass[orig_t.tag]
    oe = orig_t.attributes.values()
    ve = vm_t.attributes.values()
    tup = tuple(convert_result(getattr(res, attr), o, v)
                for attr, o, v in zip(orig_t.attributes, oe, ve))
    return dc(*tup)


@overload  # noqa: F811
def _convert_result(res, orig_t, vm_t: List):
    ot = orig_t.element_type
    vt = vm_t.element_type
    return [convert_result(x, ot, vt) for x in res]


@overload  # noqa: F811
def _convert_result(res, orig_t, vm_t: Tuple):
    # If the EraseClass opt was applied, orig_t may be Class
    orig_is_class = ismyiatype(orig_t, Class)
    if orig_is_class:
        oe = orig_t.attributes.values()
    else:
        oe = orig_t.elements
    ve = vm_t.elements
    tup = tuple(convert_result(x, o, v)
                for x, o, v in zip(res, oe, ve))
    if orig_is_class:
        dc = tag_to_dataclass[orig_t.tag]
        return dc(*tup)
    else:
        return tup


@overload  # noqa: F811
def _convert_result(arg, orig_t, vm_t: (Int, Float, Bool, Array)):
    return arg


def convert_result(res, orig_t, vm_t):
    """Convert result from vm_t to orig_t."""
    return _convert_result[vm_t](res, orig_t, vm_t)


class OutputWrapper(PipelineStep):
    """Pipeline step to convert to and from the VM's data format.

    For example, dataclasses in the arguments may be converted to
    tuples, and tuples returned by the VM would be converted back
    to the appropriate dataclasses, as determined by Myia.

    Inputs:
        graph: The root graph.
        output: The callable returned by the VM.
        argspec: Contains the original types/etc. of the arguments.
        inference_results: Contains the original return type.

    Outputs:
        output: The wrapped callable.
    """

    def step(self, graph, output, argspec, inference_results):
        """Convert args to vm format, and output from vm format."""
        fn = output
        orig_arg_t = [arg['type'] for arg in argspec]
        vm_arg_t = graph.type.arguments
        orig_out_t = inference_results['type']
        vm_out_t = graph.type.retval

        def wrapped(*args):
            args = tuple(convert_arg(arg, ot, vt) for arg, ot, vt in
                         zip(args, orig_arg_t, vm_arg_t))
            res = fn(*args)
            res = convert_result(res, orig_out_t, vm_out_t)
            return res
        return {'output': wrapped}


step_parse = Parser.partial()


step_resolve = Optimizer.partial(
    run_only_once=True,
    phases=dict(
        resolve=[optlib.resolve_globals]
    )
)


step_infer = InferrerStep.partial(
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
)


step_specialize = Specializer.partial()


step_prepare = Preparator.partial(
    erase_classes=True
)


step_opt = Optimizer.partial(
    phases=dict(
        main=[
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            optlib.simplify_always_true_switch,
            optlib.simplify_always_false_switch,
            optlib.inline_unique_uses,
            optlib.simplify_partial,
            optlib.replace_applicator,
            optlib.elim_identity,
        ],
        cse=CSE
    )
)


step_validate = Validator.partial()


step_cconv = ClosureConverter.partial()


step_debug_export = DebugVMExporter.partial(
    implementations=vm_implementations
)


step_wrap = OutputWrapper.partial()


_standard_pipeline = PipelineDefinition(
    resources=dict(
        manager=GraphManager.partial(),
        py_implementations=py_implementations,
        method_map=standard_method_map,
        convert=Converter.partial(
            object_map=standard_object_map,
            converter=default_convert
        ),
        live_infer=InferrerStep.partial(
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
            context_class=Contextless,
        )
    ),
    steps=dict(
        parse=step_parse,
        resolve=step_resolve,
        infer=step_infer,
        specialize=step_specialize,
        prepare=step_prepare,
        opt=step_opt,
        cconv=step_cconv,
        validate=step_validate,
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
        export=step_export,
        wrap=step_wrap,
    )

standard_debug_pipeline = _standard_pipeline \
    .insert_after(
        export=step_debug_export,
        wrap=step_wrap,
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
        inf = pip.steps.infer
        argspec = tuple({'value': arg} for arg in args)
        inf.fill_in(argspec)
        argnames = inspect.getargspec(self.fn).args
        for arg, name in zip(argspec, argnames):
            if name not in self.specialize_values:
                arg['value'] = ANYTHING
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
