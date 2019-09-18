"""Transforms a graph into lower-level code."""

import tvm
from tvm import relay
from tvm.relay import adt
from tvm.relay.backend import interpreter

from ...abstract import (
    AbstractArray,
    AbstractFunction,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    PartialApplication,
    TypedPrimitive,
    VirtualFunction,
)
from ...graph_utils import toposort
from ...ir import manage
from ...operations import Primitive, primitives as P
from ...xtype import Bool, Nil, type_to_np_dtype
from ..transform import get_prim_graph, wrap_result
from . import Backend, Converter, HandleBackend
from .channel import handle
from .relay_helpers import add_functions, optimize

relay_from_scalar = tvm.get_global_func('relay.from_scalar')


@wrap_result.register
def wrap_result(data: interpreter.TupleValue):
    """Wrap tuples from relay."""
    return tuple(handle(d) for d in data)


@overload(bootstrap=True)
def to_relay_type(self, a: AbstractScalar, mod):
    """Convert a myia abstract to a Relay type."""
    tp = a.xtype()
    if issubclass(tp, Bool):
        return relay.ty.TensorType((), 'bool')
    elif issubclass(tp, Nil):
        return relay.ty.TupleType([])
    else:
        return relay.ty.TensorType((), type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTuple, mod):
    return relay.ty.TupleType([self(e, mod) for e in a.elements])


@overload  # noqa: F811
def to_relay_type(self, a: AbstractArray, mod):
    tp = a.element.xtype()
    return relay.ty.TensorType(a.xshape(), type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractFunction, mod):
    sings = list(self(sing, mod) for sing in a.get_sync())
    for sing in sings[1:]:
        assert sing == sings[0]
    return sings[0]


@overload  # noqa: F811
def to_relay_type(self, a: (VirtualFunction, TypedPrimitive), mod):
    return relay.ty.FuncType([self(aa, mod) for aa in a.args],
                             self(a.output, mod))


@overload  # noqa: F811
def to_relay_type(self, a: PartialApplication, mod):
    tp = self(a.fn, mod)
    return relay.ty.FuncType(tp.arg_types[len(a.args):], tp.ret_type)


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTaggedUnion, mod):
    return mod._myia_union_type()


def ashape(node):
    """Make sure shape isn't None, that makes relay crash later."""
    shp = node.shape
    assert shp is not None
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

    P.array_to_scalar: lambda x: x,
    P.dot: lambda x, y: relay.op.nn.dense(x, relay.op.transpose(y)),

    P.make_tuple: lambda *args: relay.Tuple(args),
    P.switch: relay.If,
}


def relay_partial(c, fn, *args):
    """Implementation of partial for Relay."""
    ty = to_relay_type(fn.abstract, c.module)
    rargs = [relay.var("") for a in ty.arg_types]
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


def relay_reshape(c, v, shp):
    """Implementation of reshape for Relay."""
    nv = c.ref(v)
    assert shp.is_constant(tuple)
    trim = False
    if shp.value == ():
        shp = (1,)
        trim = True
    else:
        shp = shp.value
    res = relay.op.reshape(nv, newshape=shp)
    if trim:
        res = relay.op.take(res, relay.const(0), mode='fast')
    return res


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
            rtshp = tshp
            if tshp == ():
                tshp = (1,)
            res = relay.op.reshape(res, newshape=tshp)
            if rtshp == ():
                res = relay.op.take(res, relay.const(0))
        return res
    else:
        raise NotImplementedError(f"reduce with {fn}")


def relay_tuple_getitem(c, t, idx):
    """Implementation of tuple_getitem for Relay."""
    assert idx.is_constant(int)
    return relay.expr.TupleGetItem(c.ref(t), idx.value)


def relay_casttag(c, x, tag):
    """Implementation of casttag for Relay."""
    assert tag.is_constant(int)
    v = relay.Var("v")
    rtag = c.tag_map[tag.value]
    clause = adt.Clause(adt.PatternConstructor(rtag, [adt.PatternVar(v)]), v)
    return adt.Match(c.ref(x), [clause], complete=False)


def relay_hastag(c, x, tag):
    """Implementation of hastag for Relay."""
    assert tag.is_constant(int)
    rtag = c.tag_map[tag.value]
    v = relay.Var("v")
    t_clause = adt.Clause(adt.PatternConstructor(rtag, [adt.PatternVar(v)]),
                          relay.const(True))
    f_clause = adt.Clause(adt.PatternWildcard(), relay.const(False))
    return adt.Match(c.ref(x), [t_clause, f_clause])


def relay_tagged(c, x, tag):
    """Implementation of tagged for Relay."""
    assert tag.is_constant(int)
    rtag = c.tag_map[tag.value]
    return rtag(c.ref(x))


COMPLEX_MAP = {
    P.partial: relay_partial,
    P.distribute: relay_distribute,
    P.transpose: relay_transpose,
    P.reshape: relay_reshape,
    P.array_map: relay_array_map,
    P.array_reduce: relay_array_reduce,
    P.scalar_to_array: lambda c, x, t: c.ref(x),
    P.tuple_getitem: relay_tuple_getitem,
    P.casttag: relay_casttag,
    P.hastag: relay_hastag,
    P.tagged: relay_tagged,
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


class NodeVisitor:
    """Visitor for node enumeration."""
    def _visit_array_map(self, node):
        return node.inputs[2:]

    def _visit_array_reduce(self, node):
        return node.inputs[2:]

    def _visit_scalar_to_array(self, node):
        return [node.inputs[1]]

    def __call__(self, node):
        """Don't visit called primitives."""
        if node.inputs:
            if node.inputs[0].is_constant(Primitive):
                prim = node.inputs[0].value
                visit = getattr(self, f'_visit_{prim}', None)
                if visit is None:
                    return node.inputs[1:]
                return visit(node)
            else:
                return node.inputs
        return []


class CompileGraph:
    """Step to convert a myia graph to a relay graph.

    Inputs:
        graph: A graph

    Outputs:
        output: a wrapped relay graph
    """

    def run(self, graph, context, target):
        """Convert the graph into a relay callable."""
        mng = manage(graph)

        self.module = relay.Module({})
        self.union_type = relay.GlobalTypeVar('$_union_adt')
        self.module._myia_union_type = self.union_type
        self.build_union_adt(mng)

        # Analyze and create a global union type of all the possible types
        # and then use it for all union values.

        self.module._myia_adt_map = {}
        function_map = {}
        self.node_map = {}
        self.graph_map = {}

        for g in mng.graphs:
            if g.parent is None:
                self.graph_map[g] = relay.GlobalVar(g.debug.debug_name)

        for g in mng.graphs:
            if g.parent is None:
                function_map[self.graph_map[g]] = self.convert_func(g)

        add_functions(self.module, function_map)

        # Maybe make a function that calls the right graph instead?
        self.module["main"] = self.module[self.graph_map[graph]]

        self.module = optimize(self.module)

        exec = relay.create_executor(mod=self.module, ctx=context,
                                     target=target)
        res = exec.evaluate(self.module["main"])

        def _conv(value, type):
            if isinstance(type, AbstractTuple):
                return interpreter.TupleValue(*[_conv(e, et) for e, et in
                                                zip(value, type.elements)])
            elif isinstance(type, AbstractTaggedUnion):
                ctr = self.tag_map[value.tag]
                conv_val = _conv(value.value, type.options.get(value.tag))
                return interpreter.ConstructorValue(ctr.tag, [conv_val], None)
            elif isinstance(type, AbstractScalar):
                xt = type.xtype()
                if issubclass(xt, (Number, Bool)):
                    dtype = type_to_np_dtype(xt)
                    return relay_from_scalar(value, dtype)
                elif xt is Nil:
                    return interpreter.TupleValue()
                else:
                    raise ValueError(f"unsupported scalar {xt}")
            else:
                raise ValueError(f"unsupported {type}")

        types = [p.abstract for p in graph.parameters]
        def f(*args):
            args = tuple(_conv(arg, typ) for arg, typ in zip(args, types))
            return wrap_result(res(*args))
        return f

    def build_union_adt(self, mng):
        """Build an ADT to represent union types."""
        self.tag_map = dict()
        for node in mng.all_nodes:
            if isinstance(node.abstract, AbstractTaggedUnion):
                for opt in node.abstract.options:
                    if opt[0] not in self.tag_map:
                        t = to_relay_type(opt[1], self.module)
                        self.tag_map[opt[0]] = adt.Constructor(
                            f"c{opt[0]}", [t], self.union_type)
        self.module[self.union_type] = adt.TypeData(
            self.union_type, [], list(self.tag_map.values()))

    def on_parameter(self, node):
        """Convert a parameter node."""
        return relay.var(
            node.debug.debug_name,
            type_annotation=to_relay_type(node.abstract, self.module))

    def on_apply(self, node):
        """Convert an Apply node."""
        if node.inputs[0].is_constant(Primitive):
            fn = node.inputs[0].value
            conv = MAP.get(fn)
            if conv is not None:
                return conv(self, *node.inputs[1:])
        return relay.Call(self.ref(node.inputs[0]),
                          [self.ref(i) for i in node.inputs[1:]])

    def make_const(self, value, type):
        """Convert to a relay value."""
        if isinstance(type, AbstractTuple):
            return relay.Tuple([self.make_const(e, et) for e, et in
                                zip(value, type.elements)])
        if isinstance(type, AbstractTaggedUnion):
            ctr = self.tag_map[value.tag]
            return ctr(self.make_const(value.value,
                                       type.options.get(value.tag)))
        else:
            dtype = type_to_np_dtype(type.xtype())
            return relay.const(value, dtype=dtype)

    def on_constant(self, node):
        """Convert a constant node."""
        if node.is_constant(Primitive):
            return self.convert_func(
                get_prim_graph({}, node.value, node.abstract))
        return self.make_const(node.value, node.abstract)

    def on_graph(self, node):
        """Convert a graph constant."""
        if node.value.parent is None:
            return self.graph_map[node.value]
        if node not in self.node_map:
            self.node_map[node] = self.convert_func(node.value)
        return self.node_map[node]

    def ref(self, node):
        """Return the value for a node."""
        return self.node_map[node]

    def convert_func(self, graph):
        """Convert a graph."""
        for p in graph.parameters:
            self.node_map[p] = self.on_parameter(p)

        params = [self.ref(p) for p in graph.parameters]

        for node in toposort(graph.output, NodeVisitor()):
            if node.is_apply():
                self.node_map[node] = self.on_apply(node)
            elif node.is_constant_graph():
                self.node_map[node] = self.on_graph(node)
            elif node.is_constant():
                self.node_map[node] = self.on_constant(node)

        return relay.Function(params, self.ref(graph.output),
                              ret_type=to_relay_type(graph.output.abstract,
                                                     self.module))


compiler = CompileGraph()


class RelayBackend(ConcreteBackend):
    """Backend based on Relay.

    Backend options:
        target: the target device class ('cpu', 'cuda')
        device_id: the target device identifier (an int)
    """

    def __init__(self, target, device_id):
        """Create a Relay backend for the given device."""
        device_id = int(device_id)
        self.context = tvm.ndarray.context(target, device_id)
        if target == 'cpu':
            target = 'llvm'
        self.target = target
        if not self.context.exist:
            raise RuntimeError("No hardware to support selected target "
                               f"'{target}' on device {device_id}")
        self.compiler = compiler

    def compile(self, graph, argspec, outspec):
        """Compiler a graph."""
        return self.compiler.run(graph, self.context, self.target)

    def to_numpy(self, v):
        """Make a numpy array from a TVM array."""
        return v.asnumpy()

    def from_numpy(self, a):
        """Make an TVM array from a numpy array."""
        return tvm.ndarray.array(a, self.context)

    def to_scalar(self, v):
        """Convert the TVM array to a scalar."""
        if isinstance(v, (interpreter.TupleValue, tuple)):
            assert len(v) == 0
            return None
        else:
            return v.asnumpy().item()

    def from_scalar(self, s, t):
        """Convert the scalar to an TVM array."""
        if t == Nil:
            return ()
        dt = type_to_np_dtype(t)
        return self.from_numpy(np.array(s, dtype=dt, copy=False))


def RelayBackendR(target, device_id):
    """Relay proxy."""
    return HandleBackend(RelayBackend(target, device_id))


__all__ = [
    'RelayBackend',
    'RelayBackendR',
]
