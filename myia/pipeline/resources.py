"""Pipeline resources."""

import math
import numpy as np
from collections import defaultdict
from types import FunctionType

from .. import dtype, operations, parser, composite as C
from ..abstract.base import AbstractFunction
from ..infer import InferenceEngine, ANYTHING
from ..ir import Graph, clone
from ..prim import ops as P
from ..utils import overload, TypeMap, UNKNOWN

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
    operations.switch: P.switch,
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
    operations.switch: P.switch,
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
        '__eq__': P.bool_eq,
        '__ne__': C.bool_ne,
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
        '__myia_to_array__': P.identity,
        'item': P.array_to_scalar,
    },
    dtype.JTagged: {
    },
    dtype.SymbolicKeyType: {
    },
    dtype.EnvType: {
    },
})


#####################
# ConverterResource #
#####################


@overload
def default_convert(env, fn: FunctionType):
    """Default converter for Python types."""
    g = clone(parser.parse(fn))
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
def default_convert(env, x: (object, type, dtype.TypeMeta)):
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
        except (TypeError, KeyError):
            v = self.converter(self, value)

        if isinstance(v, Graph):
            self.resources.manager.add_graph(v)
        return v


class InferenceResource(PipelineResource):
    """Performs inference and specialization."""

    def __init__(self,
                 pipeline_init,
                 tracks,
                 required_tracks,
                 tied_tracks,
                 context_class,
                 erase_value,
                 version=1):
        """Initialize an InferenceResource."""
        super().__init__(pipeline_init)
        self.manager = self.resources.manager
        self.tracks = tracks
        self.required_tracks = required_tracks
        self.tied_tracks = tied_tracks
        self.context_class = context_class
        self.erase_value = erase_value
        self.version = version
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
            if '_erase_value' in arg:
                erase = arg['_erase_value']
                # del arg['_erase_value']
            else:
                erase = self.erase_value
            if 'value' in arg:
                v = arg['value']
                for track_name, track in self.engine.tracks.items():
                    if track_name not in arg:
                        arg[track_name] = track.from_value(v, None)
                    else:
                        arg[track_name] = track.from_external(arg[track_name])
                if erase:
                    arg['value'] = ANYTHING
            else:
                for track_name, track in self.engine.tracks.items():
                    if track_name in arg:
                        arg[track_name] = track.from_external(arg[track_name])
                    else:
                        arg[track_name] = track.default(arg)

        from ..abstract.base import broaden
        for arg in argspec:
            if '_erase_value' in arg:
                erase = arg['_erase_value']
                del arg['_erase_value']
            else:
                erase = self.erase_value
            if erase and 'abstract' in arg:
                arg['abstract'] = broaden(arg['abstract'], None)

    def infer(self, graph, argspec, outspec=None, clear=False):
        """Perform inference."""
        if clear:
            self.engine.cache.clear()
            for node in self.manager.all_nodes:
                orig_t = node.abstract
                node.inferred = defaultdict(lambda: UNKNOWN)
                if node.is_constant() \
                        and not isinstance(orig_t, AbstractFunction):
                    if orig_t is not None:
                        node.abstract = orig_t
        self.fill_in(argspec)
        return self.engine.run(
            graph,
            argspec=argspec,
            outspec=outspec,
            tracks=self.required_tracks
        )

    def specialize(self, graph, context):
        """Perform specialization."""
        if self.version == 1:
            spc = TypeSpecializer(self.engine)
        else:
            from ..aspecialize import TypeSpecializer as TypeSpecializer2
            spc = TypeSpecializer2(self.engine)
        result = spc.run(graph, context)
        self.manager.keep_roots(result)
        return result

    def renormalize(self, graph, argspec, outspec=None):
        """Perform inference and specialization."""
        _, context = self.infer(graph, argspec, outspec, clear=True)
        return self.specialize(graph, context)
