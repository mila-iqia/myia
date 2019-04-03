"""Transforms a graph into lower-level code."""

import tvm
from tvm import relay
import numpy as np

from . import Backend

from ...abstract import AbstractArray, AbstractTuple, AbstractScalar, \
    AbstractFunction, VirtualFunction, GraphFunction, TypedPrimitive, \
    PartialApplication, SHAPE, build_type
from ...ir import manage
from ...graph_utils import toposort
from ...prim import Primitive, ops as P
from ...dtype import type_to_np_dtype, ismyiatype, Bool, Tuple
from ...utils import overload

from ..transform import set_types
from .relay_helpers import optimize, build_module


@overload(bootstrap=True)
def to_relay_type(self, a: AbstractScalar):
    """Convert a myia abstract to a Relay type."""
    tp = build_type(a)
    if ismyiatype(tp, Bool):
        return relay.ty.TensorType((), 'bool')
    else:
        return relay.ty.TensorType((), type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTuple):
    return relay.ty.TupleType([self(e) for e in a.elements])


@overload  # noqa: F811
def to_relay_type(self, a: AbstractArray):
    tp = build_type(a.element)
    return relay.ty.TensorType(a.values[SHAPE], type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractFunction):
    return self(a.get_unique())


@overload  # noqa: F811
def to_relay_type(self, a: (VirtualFunction, TypedPrimitive)):
    return relay.ty.FuncType([self(aa) for aa in a.args],
                             self(a.output))


@overload  # noqa: F811
def to_relay_type(self, a: PartialApplication):
    tp = self(a.fn)
    return relay.ty.FuncType(tp.arg_types[len(a.args):], tp.ret_type)


@overload  # noqa: F811
def to_relay_type(self, a: GraphFunction):
    return self(a.graph.abstract)


@overload  # noqa: F811
def to_relay_type(self, a: object):  # pragma: no cover
    raise ValueError("Unknown type:", build_type(a))


def ashape(node):
    """Make sure shape isn't None, that makes relay crash later."""
    shp = node.shape
    if shp is None:  # pragma: no cover
        raise RuntimeError("You found a way to trigger this, please report it")
        shp = ()
    return shp


SIMPLE_MAP = {
    P.scalar_add: relay.op.add,
    P.scalar_sub: relay.op.subtract,
    P.scalar_mul: relay.op.multiply,
    P.scalar_div: relay.op.divide,
    P.scalar_mod: relay.op.mod,
    P.scalar_pow: relay.op.power,
    P.scalar_floor: relay.op.floor,
    P.scalar_uadd: lambda x: x,
    P.scalar_usub: relay.op.negative,
    P.scalar_exp: relay.op.exp,
    P.scalar_log: relay.op.log,
    # This is not right tangent vs hyperbolic tangent
    # P.scalar_tan: relay.op.tanh,
    P.scalar_tanh: relay.op.tanh,

    P.scalar_eq: relay.op.equal,
    P.scalar_lt: relay.op.less,
    P.scalar_gt: relay.op.greater,
    P.scalar_ne: relay.op.not_equal,
    P.scalar_le: relay.op.less_equal,
    P.scalar_ge: relay.op.greater_equal,
    P.bool_and: relay.op.logical_and,
    P.bool_or: relay.op.logical_or,
    P.bool_eq: relay.op.equal,
    P.bool_not: relay.op.logical_not,

    P.scalar_to_array: lambda x: x,
    P.dot: lambda x, y: relay.op.nn.dense(x, relay.op.transpose(y)),

    P.make_tuple: lambda *args: relay.Tuple(args),
    P.switch: relay.If,
}


def relay_partial(c, fn, *args):
    """Implementation of partial for Relay."""
    ty = fn.type
    rargs = [relay.var("") for a in ty.arguments]
    fn = relay.Function(rargs, relay.Call(c.ref(fn), rargs))
    binds = {}
    for ra, a in zip(rargs, args):
        binds[ra] = c.ref(a)
    res = relay.bind(fn, binds)
    return res


def relay_distribute(c, array, shape):
    """Implementation of distribute for Relay."""
    assert shape.is_constant(tuple)
    return relay.op.broadcast_to(c.ref(array), shape.value)


def relay_transpose(c, a, ax):
    """Implementation of transpose for Relay."""
    na = c.ref(a)
    assert ax.is_constant(tuple)
    return relay.op.transpose(na, axes=ax.value)


def relay_array_map(c, fn, *array):
    """Implementation of array_map for Relay."""
    assert fn.is_constant(Primitive)
    fn = fn.value
    return SIMPLE_MAP[fn](*[c.ref(a) for a in array])


def relay_array_reduce(c, fn, array, shape):
    """Implementation of array_reduce for Relay."""
    assert fn.is_constant(Primitive)
    assert shape.is_constant(tuple)
    fn = fn.value
    tshp = shape.value
    ary = c.ref(array)
    if fn == P.scalar_add:
        ashp = ashape(array)
        if len(tshp) < len(ashp):
            ts = (1,) * (len(ashp) - len(tshp)) + tshp
        else:
            ts = tshp
        axis = tuple(i for i, t in enumerate(ts) if t == 1)
        res = relay.op.sum(ary, axis=axis, keepdims=True)
        if len(tshp) < len(ashp):
            res = relay.op.reshape(res, newshape=tshp)
        return res
    else:
        raise NotImplementedError(f"reduce with {fn}")


COMPLEX_MAP = {
    P.partial: relay_partial,
    P.distribute: relay_distribute,
    P.transpose: relay_transpose,
    P.array_map: relay_array_map,
    P.array_reduce: relay_array_reduce
}


class RelayMapper:
    """Maps myia operations to relay operations."""

    def __init__(self, simple_map=None, complex_map=None):
        """Create a mapper."""
        self.mapping = {}
        if simple_map is not None:
            self.register_simple(simple_map)
        if complex_map is not None:
            self.register_complex(complex_map)

    def register(self, prim, fn):
        """Register the conversion function for a primitve."""
        assert prim not in self.mapping
        self.mapping[prim] = fn

    def register_simple(self, map):
        """Register simple conversions (1:1 map to nnvm ops)."""
        for k, v in map.items():
            self.register(k, lambda c, *args, v=v: v(*[c.ref(a)
                                                       for a in args]))

    def register_complex(self, map):
        """Register complex conversions."""
        for k, v in map.items():
            self.register(k, v)

    def get(self, fn):
        """Get the mapping for the primitive."""
        return self.mapping.get(fn, None)


MAP = RelayMapper(simple_map=SIMPLE_MAP, complex_map=COMPLEX_MAP)


class CompileGraph:
    """Step to convert a myia graph to a relay graph.

    Inputs:
        graph: A graph

    Outputs:
        output: a wrapped relay graph
    """

    def run(self, graph, context):
        """Convert the graph into a relay callable."""
        mng = manage(graph)

        function_map = {}
        self.node_map = {}
        self.graph_map = {}

        for g in mng.graphs:
            self.graph_map[g] = relay.GlobalVar(g.debug.debug_name)

        for g in mng.graphs:
            function_map[self.graph_map[g]] = self.convert_func(g)

        module = build_module(function_map)

        module.entry_func = module.global_var_map_[graph.debug.debug_name]

        optimize(module)

        exec = relay.create_executor(mod=module, ctx=context)
        return exec.evaluate(module.entry_func)

    def on_parameter(self, node):
        """Convert a parameter node."""
        return relay.var(
            node.debug.debug_name,
            type_annotation=to_relay_type(node.abstract))

    def on_apply(self, node):
        """Convert an Apply node."""
        if node.inputs[0].is_constant(Primitive):
            fn = node.inputs[0].value
            conv = MAP.get(fn)
            if conv is not None:
                return conv(self, *node.inputs[1:])
        return relay.Call(self.ref(node.inputs[0]),
                          [self.ref(i) for i in node.inputs[1:]])

    def on_constant(self, node):
        """Convert a constant node."""
        def _helper(value, type):
            if ismyiatype(type, Tuple):
                return relay.Tuple([_helper(e, et) for e, et in
                                    zip(value, type.elements)])
            else:
                return relay.const(value,
                                   dtype=type_to_np_dtype(type))
        if isinstance(node.value, Primitive):
            # This is a hack for list_map and friends
            return None
        return _helper(node.value, node.type)

    def on_graph(self, node):
        """Convert a graph constant."""
        return self.graph_map[node.value]

    def ref(self, node):
        """Return the value for a node."""
        return self.node_map[node]

    def convert_func(self, graph):
        """Convert a graph."""
        for p in graph.parameters:
            self.node_map[p] = self.on_parameter(p)

        params = [self.ref(p) for p in graph.parameters]

        def visit_noprimfunc(node):
            """Don't visit called primitives."""
            if node.inputs:
                if node.inputs[0].is_constant(Primitive):
                    return node.inputs[1:]
                else:
                    return node.inputs
            return []

        for node in toposort(graph.output, visit_noprimfunc):
            if node.is_apply():
                self.node_map[node] = self.on_apply(node)
            elif node.is_constant_graph():
                self.node_map[node] = self.on_graph(node)
            elif node.is_constant():
                self.node_map[node] = self.on_constant(node)

        return relay.Function(params, self.ref(graph.output),
                              ret_type=to_relay_type(graph.output.abstract))


compiler = CompileGraph()


class RelayBackend(Backend):
    """Backend based on Relay.

    Backend options:
        target: the target device class ('cpu', 'cuda')
        device_id: the target device identifier (an int)
    """

    def __init__(self, target='cpu', device_id=0):
        """Create a Relay backend for the given device."""
        self.context = tvm.ndarray.context(target, device_id)
        if not self.context.exist:  # pragma: no cover
            raise RuntimeError("No hardware to support selected target/device")
        self.compiler = compiler

    def compile(self, graph, argspec, outspec, pipeline):
        """Compiler a graph."""
        graph = set_types(graph, argspec, outspec, pipeline)
        return self.compiler.run(graph, self.context)

    def to_numpy(self, v):
        """Make a numpy array from a TVM array."""
        return v.asnumpy()

    def from_numpy(self, a):
        """Make an TVM array from a numpy array."""
        return tvm.ndarray.array(a, self.context)

    def to_scalar(self, v):
        """Convert the TVM array to a scalar."""
        return v.asnumpy().item()

    def from_scalar(self, s, t):
        """Convert the scalar to an TVM array."""
        dt = type_to_np_dtype(t)
        return self.from_numpy(np.array(s, dtype=dt, copy=False))

    def to_dlpack(self, v):
        """Make a dlpack capsule from an TVM array."""
        return v.to_dlpack()

    def from_dlpack(self, v):
        """Make an TVM array from a dlpack capsule."""
        t = tvm.ndarray.from_dlpack(v)
        if t.context != self.context:  # pragma: no cover
            # This may do a copy but we will need it
            t = tvm.ndarray.array(t, self.context)
        return t

    def check_array(self, v, t):
        """Check if value is an TVM array for this context."""
        if not isinstance(v, tvm.ndarray.NDArray):
            raise TypeError("Expected NNVM array")
        if v.context != self.context:  # pragma: no cover
            raise RuntimeError("Array on wrong context.")
        if v.dtype != type_to_np_dtype(t):
            raise TypeError("Wrong dtype")
