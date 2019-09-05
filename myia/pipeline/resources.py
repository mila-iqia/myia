"""Pipeline resources."""

import math
from types import FunctionType

import numpy as np

from .. import composite as C, macros as M, operations, parser, xtype
from ..abstract import InferenceEngine
from ..ir import Graph, clone
from ..monomorphize import monomorphize
from ..prim import ops as P
from ..utils import Slice, TypeMap, overload, tracer
from .pipeline import PipelineResource

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
    operations.is_: C.is_,
    operations.is_not: C.is_not,
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
    operations.getattr: M.getattr_,
    operations.setattr: P.record_setitem,
    operations.len: C._len,
    operations.make_tuple: P.make_tuple,
    operations.make_list: M.make_list,
    operations.make_dict: P.make_dict,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.resolve: M.resolve,
    operations.to_array: C.to_array,
    operations.switch: P.switch,
    operations.user_switch: M.user_switch,
    operations.slice: Slice,
    operations.apply: M.apply,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: P.scalar_exp,
    math.log: P.scalar_log,
    math.sin: P.scalar_sin,
    math.cos: P.scalar_cos,
    math.tan: P.scalar_tan,
    sum: C.sum,
    Exception: P.exception,
    range: C.range_,
    zip: C.zip_,
    enumerate: C.enumerate_,
    isinstance: M.isinstance_,
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
    operations.is_: C.is_,
    operations.is_not: C.is_not,
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
    operations.getattr: M.getattr_,
    operations.setattr: P.record_setitem,
    operations.len: C._len,
    operations.make_tuple: P.make_tuple,
    operations.make_list: M.make_list,
    operations.make_dict: P.make_dict,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.next: C.next,
    operations.resolve: M.resolve,
    operations.to_array: C.to_array,
    operations.switch: P.switch,
    operations.user_switch: M.user_switch,
    operations.slice: Slice,
    operations.apply: M.apply,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: P.scalar_exp,
    math.log: P.scalar_log,
    math.sin: P.scalar_sin,
    math.cos: P.scalar_cos,
    math.tan: P.scalar_tan,
    math.tanh: P.scalar_tanh,
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
    np.tanh: C.tanh,
    np.sum: C.sum,
    sum: C.sum,
    Exception: P.exception,
    range: C.range_,
    zip: C.zip_,
    enumerate: C.enumerate_,
    isinstance: M.isinstance_,
}

standard_method_map = TypeMap({
    xtype.Nil: {
        '__eq__': C.nil_eq,
        '__ne__': C.nil_ne,
        '__bool__': C.nil_bool,
    },
    xtype.Bool: {
        '__and__': P.bool_and,
        '__or__': P.bool_or,
        '__eq__': P.bool_eq,
        '__ne__': C.bool_ne,
        '__bool__': P.identity,
    },
    xtype.String: {
        '__eq__': P.string_eq,
        '__ne__': C.string_ne,
    },
    xtype.Int: {
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
    xtype.UInt: {
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
    xtype.Float: {
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
    xtype.Tuple: {
        '__len__': P.tuple_len,
        '__add__': C.tuple_concat,
        '__getitem__': C.tuple_get,
        '__setitem__': P.tuple_setitem,
        '__myia_iter__': P.identity,
        '__myia_next__': C.tuple_next,
        '__myia_hasnext__': C.tuple_hasnext,
    },
    xtype.Dict: {
        '__getitem__': P.dict_getitem,
        'values': M.dict_values,
    },
    xtype.NDArray: {
        '__add__': C.add,
        '__sub__': C.sub,
        '__mul__': C.mul,
        '__truediv__': C.truediv,
        '__floordiv__': C.floordiv,
        '__mod__': C.mod,
        '__pow__': C.pow,
        '__floor__': C.array_floor,
        '__trunc__': C.array_trunc,
        '__pos__': C.array_uadd,
        '__neg__': C.array_usub,
        '__eq__': C.eq,
        '__ne__': C.ne,
        '__lt__': C.lt,
        '__gt__': C.gt,
        '__le__': C.le,
        '__ge__': C.ge,
        '__matmul__': P.dot,
        '__len__': P.array_len,
        '__getitem__': P.array_getitem,
        '__setitem__': P.array_setitem,
        '__myia_iter__': C.array_iter,
        '__myia_to_array__': P.identity,
        'item': P.array_to_scalar,
        'shape': property(P.shape),
        'T': property(C.transpose),
        'ndim': property(C.ndim),
    },
    xtype.SymbolicKeyType: {
    },
    xtype.EnvType: {
    },
})


#####################
# ConverterResource #
#####################


@overload
def default_convert(env, fn: FunctionType):
    """Default converter for Python types."""
    g = parser.parse(fn)
    if isinstance(g, Graph):
        g = clone(g)
    env.object_map[fn] = g
    return g


@overload  # noqa: F811
def default_convert(env, g: Graph):
    mng = env.resources.manager
    if g._manager is not mng:
        g2 = clone(g)
        env.object_map[g] = g2
        return g2
    else:
        return g


@overload  # noqa: F811
def default_convert(env, seq: (tuple, list)):
    return type(seq)(env(x) for x in seq)


@overload  # noqa: F811
def default_convert(env, x: (object, type, xtype.TypeMeta)):
    return x


class _Unconverted:
    # This is just used by Converter to delay conversion of graphs associated
    # to operators or methods until they are actually needed.
    def __init__(self, value):
        self.value = value


class ConverterResource(PipelineResource):
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
            bool: xtype.Bool,
            int: xtype.Int,
            float: xtype.Float,
            np.ndarray: xtype.NDArray,
            np.int8: xtype.Int,
            np.int16: xtype.Int,
            np.int32: xtype.Int,
            np.int64: xtype.Int,
            np.float16: xtype.Float,
            np.float32: xtype.Float,
            np.float64: xtype.Float,
            tuple: xtype.Tuple,
            dict: xtype.Dict,
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
        except (TypeError, KeyError):
            v = self.converter(self, value)

        if isinstance(v, Graph):
            self.resources.manager.add_graph(v)
        return v


class InferenceResource(PipelineResource):
    """Performs inference and monomorphization."""

    def __init__(self,
                 pipeline_init,
                 constructors,
                 context_class):
        """Initialize an InferenceResource."""
        super().__init__(pipeline_init)
        self.manager = self.resources.manager
        self.context_class = context_class
        self.constructors = constructors
        self.engine = InferenceEngine(
            self.pipeline,
            constructors=self.constructors,
            context_class=self.context_class,
        )

    def infer(self, graph, argspec, outspec=None, clear=False):
        """Perform inference."""
        with tracer('infer',
                    graph=graph,
                    argspec=argspec,
                    outspec=outspec) as tr:
            if clear:
                self.engine.reset()
            rval = self.engine.run(
                graph,
                argspec=tuple(arg['abstract'] if isinstance(arg, dict)
                              else arg for arg in argspec),
                outspec=outspec,
            )
            tr.set_results(output=rval)
            return rval

    def monomorphize(self, context):
        """Perform monomorphization."""
        with tracer('monomorphize',
                    engine=self.engine,
                    context=context) as tr:
            rval = monomorphize(self.engine, context, reuse_existing=True)
            tr.set_results(output=rval)
            return rval

    def renormalize(self, graph, argspec, outspec=None):
        """Perform inference and specialization."""
        _, context = self.infer(graph, argspec, outspec, clear=True)
        return self.monomorphize(context)
