"""Pipeline resources."""

import math
from types import FunctionType

import numpy as np

from .. import composite as C, macros as M, operations, parser, xtype
from ..abstract import InferenceEngine
from ..compile import load_backend
from ..ir import Graph, clone
from ..monomorphize import monomorphize
from ..prim import ops as P
from ..utils import Partial, Partializable, Slice, overload, tracer
from ..vm import VM

scalar_object_map = {
    operations.add: P.scalar_add,
    operations.sub: P.scalar_sub,
    operations.mul: P.scalar_mul,
    operations.truediv: C.truediv,
    operations.floordiv: C.floordiv,
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
    operations.getitem: P.tuple_getitem,
    operations.setitem: P.tuple_setitem,
    operations.bool: P.identity,
    operations.make_tuple: P.make_tuple,
    operations.hastype: P.hastype,
    operations.switch: P.switch,
    operations.user_switch: P.switch,
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
    operations.is_: M.is_,
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
    operations.hasattr: M.hasattr_,
    operations.setattr: P.record_setitem,
    operations.len: C._len,
    operations.make_tuple: P.make_tuple,
    operations.make_list: M.make_list,
    operations.make_dict: P.make_dict,
    operations.embed: M.embed,
    operations.iter: C.iter,
    operations.hasnext: C.hasnext,
    operations.hastype: P.hastype,
    operations.next: C.next,
    operations.resolve: M.resolve,
    operations.to_array: C.to_array,
    operations.typeof: M.typeof,
    operations.switch: P.switch,
    operations.user_switch: M.user_switch,
    operations.slice: Slice,
    operations.apply: M.apply,
    operations.J: P.J,
    operations.Jinv: P.Jinv,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: P.scalar_exp,
    math.log: P.scalar_log,
    math.sin: P.scalar_sin,
    math.cos: P.scalar_cos,
    math.tan: P.scalar_tan,
    math.tanh: P.scalar_tanh,
    np.floor: C.array_floor,
    np.trunc: C.array_trunc,
    np.add: C.array_add,
    np.subtract: C.array_sub,
    np.multiply: C.array_mul,
    np.divide: C.array_truediv,
    np.mod: C.array_mod,
    np.power: C.array_pow,
    np.exp: C.array_exp,
    np.log: C.array_log,
    np.sin: C.array_sin,
    np.cos: C.array_cos,
    np.tan: C.array_tan,
    np.tanh: C.array_tanh,
    np.sum: C.sum,
    sum: C.sum,
    Exception: P.exception,
    range: C.range_,
    zip: C.zip_,
    enumerate: C.enumerate_,
    isinstance: M.isinstance_,
}


standard_method_map = {
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
    xtype.Number: {
        '__add__': P.scalar_add,
        '__sub__': P.scalar_sub,
        '__mul__': P.scalar_mul,
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
    xtype.Int: {
        '__floordiv__': C.int_floordiv,
        '__truediv__': C.int_truediv,
        '__floor__': P.identity,
        '__trunc__': P.identity,
        '__bool__': C.int_bool,
        '__myia_to_array__': P.scalar_to_array,
    },
    xtype.UInt: {
        '__floordiv__': P.scalar_div,
        '__truediv__': C.int_truediv,
        '__floor__': P.identity,
        '__trunc__': P.identity,
        '__bool__': C.int_bool,
        '__myia_to_array__': P.scalar_to_array,
    },
    xtype.Float: {
        '__floordiv__': C.float_floordiv,
        '__truediv__': P.scalar_div,
        '__floor__': P.scalar_floor,
        '__trunc__': P.scalar_trunc,
        '__bool__': C.float_bool,
        '__myia_to_array__': P.scalar_to_array,
    },
    xtype.Tuple: {
        '__len__': M.tuple_len,
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
        '__len__': M.array_len,
        '__getitem__': P.array_getitem,
        '__setitem__': P.array_setitem,
        '__myia_iter__': C.array_iter,
        '__myia_to_array__': P.identity,
        'item': P.array_to_scalar,
        'shape': property(P.shape),
        'T': property(C.transpose),
        'ndim': property(C.ndim),
    },
}


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


class ConverterResource(Partializable):
    """Convert a Python object into an object that can be in a Myia graph."""

    def __init__(self, resources, object_map):
        """Initialize a Converter."""
        self.resources = resources
        self.object_map = {}
        for k, v in object_map.items():
            self.object_map[k] = _Unconverted(v)
        for prim, impl in self.resources.py_implementations.items():
            self.object_map[impl] = prim

    def __call__(self, value):
        """Convert a value."""
        try:
            v = self.object_map[value]
            if isinstance(v, _Unconverted):
                v = default_convert(self, v.value)
                self.object_map[value] = v
        except (TypeError, KeyError):
            v = default_convert(self, value)

        if isinstance(v, Graph):
            self.resources.manager.add_graph(v)
        return v


class InferenceResource(Partializable):
    """Performs inference and monomorphization."""

    def __init__(self, resources, constructors, context_class):
        """Initialize an InferenceResource."""
        self.manager = resources.manager
        self.context_class = context_class
        self.constructors = constructors
        self.engine = InferenceEngine(
            resources,
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


class NumpyChecker:
    """Dummy backend used for debug mode."""

    def to_backend_value(self, v, t):
        """Returns v."""
        return v

    def from_backend_value(self, v, t):
        """Returns v."""
        return v


class BackendResource(Partializable):
    """Contains the backend."""

    def __init__(self, resources, name=None, options=None):
        """Initialize a BackendResource.

        Arguments:
            name: (str) the name of the backend to use
            options: (dict) options for the backend

        """
        self.resources = resources
        if name is False:
            self.backend = NumpyChecker()
        else:
            self.backend = load_backend(name, options)

    def compile(self, graph, argspec, outspec):
        """Compile the graph."""
        return self.backend.compile(graph, argspec, outspec)


class DebugVMResource(Partializable):
    """Contains the DebugVM."""

    def __init__(self, resources, implementations):
        """Initialize a DebugVMResource."""
        self.vm = VM(resources.convert,
                     resources.manager,
                     resources.py_implementations,
                     implementations)


class Resources(Partializable):
    """Defines a set of resources shared by the steps of a Pipeline."""

    def __init__(self, **members):
        """Initialize the Resources."""
        self._members = members
        self._inst = {}

    def __getattr__(self, attr):
        if attr in self._inst:
            return self._inst[attr]

        if attr in self._members:
            inst = self._members[attr]
            if isinstance(inst, Partial):
                try:
                    inst = inst.partial(resources=self)
                except TypeError:
                    pass
                inst = inst()
            self._inst[attr] = inst
            return inst

        raise AttributeError(f'No resource named {attr}.')

    def __call__(self):
        """Run the Resources as a pipeline step."""
        return {'resources': self}
