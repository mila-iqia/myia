"""Transforms a graph into lower-level code."""

from itertools import accumulate

import tvm
from tvm import relay
from tvm.relay import adt
from tvm.relay.backend import interpreter

from ...abstract import AbstractTaggedUnion
from ...graph_utils import toposort
from ...ir import manage
from ...operations import Primitive, primitives as P
from ...utils import HandleInstance, TaggedValue, new_universe
from ...xtype import UniverseType, type_to_np_dtype
from ..channel import handle
from ..transform import get_prim_graph, return_handles, wrap_result
from . import Backend, Converter, HandleBackend
from .relay_helpers import (
    TypeHelper,
    add_functions,
    dead_value,
    empty_env,
    get_myia_tag,
    get_union_ctr,
    handle_wrapper,
    optimize,
    to_relay_type,
)

relay_from_scalar = tvm.get_global_func('relay.from_scalar')


@wrap_result.register
def wrap_result(data: interpreter.TupleValue):
    """Wrap tuples from relay."""
    return tuple(handle(d) for d in data)


def ashape(node):
    """Make sure shape isn't None, that makes relay crash later."""
    shp = node.shape
    assert shp is not None
    return shp


SIMPLE_MAP = {
    P.scalar_add: relay.add,
    P.scalar_sub: relay.subtract,
    P.scalar_mul: relay.multiply,
    P.scalar_div: relay.divide,
    P.scalar_mod: relay.mod,
    P.scalar_pow: relay.op.power,
    P.scalar_floor: relay.op.floor,
    P.scalar_uadd: lambda x: x,
    P.scalar_usub: relay.op.negative,
    P.scalar_exp: relay.exp,
    P.scalar_log: relay.log,
    P.scalar_max: relay.maximum,
    P.scalar_tanh: relay.op.tanh,
    P.scalar_sign: relay.sign,
    P.scalar_abs: relay.abs,

    P.scalar_eq: relay.op.equal,
    P.scalar_lt: relay.op.less,
    P.scalar_gt: relay.op.greater,
    P.scalar_ne: relay.op.not_equal,
    P.scalar_le: relay.op.less_equal,
    P.scalar_ge: relay.op.greater_equal,
    P.bool_and: relay.op.logical_and,
    P.scalar_bit_lshift: relay.op.left_shift,
    P.scalar_bit_rshift: relay.op.right_shift,
    P.bool_or: relay.op.logical_or,
    P.bool_eq: relay.op.equal,
    P.bool_not: relay.op.logical_not,

    P.array_to_scalar: lambda x: x,
    P.dot: lambda x, y: relay.op.nn.dense(x, relay.op.transpose(y)),

    P.make_tuple: lambda *args: relay.Tuple(args),
    P.switch: relay.If,
}


def relay_distribute(c, array, shape):
    """Implementation of distribute for Relay."""
    assert shape.is_constant(tuple)
    # Make sure shape is a tuple of builtin Python integers.
    relay_shape = tuple(int(dim) for dim in shape.value)
    return relay.op.broadcast_to(c.ref(array), relay_shape)


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
    if fn is P.switch:
        rfn = relay.where
    else:
        rfn = SIMPLE_MAP[fn]
    return rfn(*[c.ref(a) for a in array])


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
    elif fn == P.scalar_mul:
        ashp = ashape(array)
        if len(tshp) in (0, len(ashp)):
            res = relay.op.prod(ary)
        else:
            raise NotImplementedError(
                'We currently support only full product on an array.')
        return res
    else:
        raise NotImplementedError(f"reduce with {fn}")


def relay_cast(c, v, t):
    """Implementation of scalar_cast/array_cast for Relay."""
    v = c.ref(v)
    assert t.is_constant()
    return relay.cast(v, type_to_np_dtype(t.value.xtype()))


def relay_tuple_getitem(c, t, idx):
    """Implementation of tuple_getitem for Relay."""
    assert idx.is_constant(int)
    return relay.expr.TupleGetItem(c.ref(t), idx.value)


def relay_casttag(c, x, tag):
    """Implementation of casttag for Relay."""
    assert tag.is_constant(int)
    rtag = get_union_ctr(tag.value, x.abstract.options.get(tag.value))
    v = relay.Var("v")
    clause = adt.Clause(adt.PatternConstructor(rtag, [adt.PatternVar(v)]), v)
    return adt.Match(c.ref(x), [clause], complete=False)


def relay_hastag(c, x, tag):
    """Implementation of hastag for Relay."""
    assert tag.is_constant(int)
    rtag = get_union_ctr(tag.value, x.abstract.options.get(tag.value))
    t_clause = adt.Clause(adt.PatternConstructor(
        rtag, [adt.PatternWildcard()]), relay.const(True))
    f_clause = adt.Clause(adt.PatternWildcard(), relay.const(False))
    return adt.Match(c.ref(x), [t_clause, f_clause])


def relay_tagged(c, x, tag):
    """Implementation of tagged for Relay."""
    assert tag.is_constant(int)
    rtag = get_union_ctr(tag.value, None)
    return rtag(c.ref(x))


def relay_env_setitem(c, env, key, x):
    """Implementation of env_setitem for Relay."""
    gv = c.types.get_env_update(x.abstract)
    return relay.Call(gv, [c.ref(env), c.ref(key), c.ref(x)])


def relay_env_getitem(c, env, key, dft):
    """Implementation of env_getitem for Relay."""
    gv = c.types.get_env_find(dft.abstract)
    return relay.Call(gv, [c.ref(env), c.ref(key), c.ref(dft)])


def relay_unsafe_static_cast(c, val, ty):
    """Implementation of unsafe_static_cast for Relay."""
    assert ty.is_constant(AbstractTaggedUnion)
    assert isinstance(val.abstract, AbstractTaggedUnion)
    return c.ref(val)


def relay_array_getitem(c, a, start, stop, strides):
    """Implementation of array_getitem for Relay."""
    assert start.is_constant(tuple)
    assert stop.is_constant(tuple)
    assert strides.is_constant(tuple)
    return relay.op.transform.strided_slice(c.ref(a), start.value, stop.value,
                                            strides.value)


def relay_argmax(c, v, dims):
    """Implementation of argmax for Relay."""
    v = c.ref(v)
    assert dims.is_constant(tuple)
    return relay.cast(relay.argmax(v, axis=dims.value), 'int64')


def relay_max_pool2d(c, img, psize, stride, pad, dil, ceil_mode):
    assert psize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert ceil_mode.is_constant(bool)
    assert dil.value == (1, 1)

    return relay.nn.max_pool2d(c.ref(img), psize.value, stride.value,
                               pad.value, ceil_mode=ceil_mode.value)


def relay_max_pool2d_grad(c, img, psize, stride, pad, dil, ceil_mode, dout):
    assert psize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert ceil_mode.is_constant(bool)
    assert dil.value == (1, 1)

    return relay.nn.max_pool2d_grad(c.ref(dout), c.ref(img), psize.value,
                                    stride.value, pad.value,
                                    ceil_mode=ceil_mode.value)


def relay_array_max(c, a, dim):
    assert dim.is_constant(tuple)
    return relay.max(c.ref(a), axis=dim.value)


def relay_conv2d(c, img, w, stride, pad, dil, groups):
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert groups.is_constant(int)

    return relay.nn.conv2d(c.ref(img), c.ref(w), strides=stride.value,
                           padding=pad.value, dilation=dil.value,
                           groups=groups.value)


def relay_conv2d_weight_grad(c, data, wsize, dout, stride, pad, dil, groups):
    assert wsize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert groups.is_constant(int)

    batch, in_channel, in_h, in_w = data.abstract.xshape()
    out_channel, _, filter_h, filter_w = wsize.value
    _, _, grad_h, grad_w = dout.abstract.xshape()
    pad_h, pad_w = pad.value

    data = c.ref(data)
    dout = c.ref(dout)

    fpad_h = pad_h * 2
    fpad_w = pad_w * 2
    fpad_top = (pad_h + 1) // 2
    fpad_left = (pad_w + 1) // 2
    fpad_bottom = fpad_h - fpad_top
    fpad_right = fpad_w - fpad_left

    padded_weight_grad_h = ((in_h - (grad_h - 1) * stride.value[0] - 1 +
                             fpad_top + fpad_bottom) // dil.value[0] + 1)
    padded_weight_grad_w = ((in_w - (grad_w - 1) * stride.value[1] - 1 +
                             fpad_left + fpad_right) // dil.value[1] + 1)

    dout = relay.tile(dout, [1, in_channel // groups.value, 1, 1])
    dout = relay.reshape(dout, [-1, 1, 0, 0])
    data = relay.reshape(data, [1, -1, 0, 0])

    d = relay.nn.conv2d(data, dout, strides=dil.value, padding=pad.value,
                        dilation=stride.value, groups=batch * in_channel)
    d = relay.reshape(d, [batch, in_channel // groups.value, out_channel,
                          padded_weight_grad_h, padded_weight_grad_w])
    d = relay.sum(d, axis=0)
    d = relay.transpose(d, [1, 0, 2, 3])
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        d = relay.strided_slice(d, begin=[0, 0, 0, 0],
                                end=[None, None, filter_h, filter_w])
    return d


def relay_conv2d_input_grad(c, isize, w, dout, stride, pad, dil, groups):
    assert isize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert groups.is_constant(int)
    if stride.value != (1, 1):
        raise ValueError("non unit stride is not supported for input grad "
                         "for now, it gives bad values")

    weight = c.ref(w)
    grad = c.ref(dout)

    data_shape = isize.value
    weight_shape = w.abstract.xshape()
    _, _, grad_h, grad_w = dout.abstract.xshape()
    bactch, in_channels, in_h, in_w = data_shape
    out_channels, _, filter_h, filter_w = weight_shape

    out_h = ((grad_h - 1) * stride.value[0] - pad.value[0] * 2 +
             (filter_h - 1) * dil.value[0] + 1)
    out_w = ((grad_w - 1) * stride.value[1] - pad.value[1] * 2 +
             (filter_w - 1) * dil.value[1] + 1)
    output_padding = (isize.value[2] - out_h, isize.value[3] - out_w)

    return relay.nn.conv2d_transpose(grad, weight,
                                     strides=stride.value,
                                     padding=pad.value, dilation=dil.value,
                                     groups=groups.value,
                                     output_padding=output_padding,
                                     kernel_size=(filter_h, filter_w),
                                     channels=in_channels)


def relay_concat(c, x, dim):
    assert dim.is_constant(int)

    xr = c.ref(x)
    inputs = [relay.expr.TupleGetItem(xr, i)
              for i in range(len(x.abstract.elements))]
    return relay.concatenate(inputs, dim.value)


def relay_split(c, x, sections, dim):
    assert sections.is_constant(tuple)
    assert dim.is_constant(int)

    sections = tuple(accumulate(sections.value))[:-1]
    return relay.split(c.ref(x), sections, dim.value).astuple()


def relay_handle(c, v):
    return relay.expr.RefCreate(c.ref(v))


# Proper sequencing is handled in convert_func() below
def relay_universe_setitem(c, u, h, v):
    return relay.RefWrite(c.ref(h), c.ref(v))


def relay_universe_getitem(c, u, h):
    return relay.RefRead(c.ref(h))


COMPLEX_MAP = {
    P.distribute: relay_distribute,
    P.transpose: relay_transpose,
    P.reshape: relay_reshape,
    P.array_map: relay_array_map,
    P.array_reduce: relay_array_reduce,
    P.scalar_to_array: lambda c, x, t: c.ref(x),
    P.array_cast: relay_cast,
    P.scalar_cast: relay_cast,
    P.tuple_getitem: relay_tuple_getitem,
    P.casttag: relay_casttag,
    P.hastag: relay_hastag,
    P.tagged: relay_tagged,
    P.env_setitem: relay_env_setitem,
    P.env_getitem: relay_env_getitem,
    P.unsafe_static_cast: relay_unsafe_static_cast,
    P.array_getitem: relay_array_getitem,
    P.argmax: relay_argmax,
    P.max_pool2d: relay_max_pool2d,
    P.max_pool2d_grad: relay_max_pool2d_grad,
    P.array_max: relay_array_max,
    P.conv2d: relay_conv2d,
    P.conv2d_weight_grad: relay_conv2d_weight_grad,
    P.conv2d_input_grad: relay_conv2d_input_grad,
    P.concat: relay_concat,
    P.split: relay_split,
    P.handle: relay_handle,
    P.universe_setitem: relay_universe_setitem,
    P.universe_getitem: relay_universe_getitem,
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
        """Register simple conversions (1:1 map to relay ops)."""
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

    def _visit_unsafe_static_cast(self, node):
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


class RelayConstantConverter(Converter):
    """Convert values to Relay constants."""

    def __init__(self, context):
        """Set the context."""
        self.context = context

    def convert_array(self, v, t):  # pragma: no cover
        """Make a TVM array from a numpy array."""
        return relay.const(tvm.ndarray.array(v, self.context))

    def convert_scalar(self, v, t):
        """Convert the scalar to a TVM array."""
        return relay.const(v, type_to_np_dtype(t))

    def convert_bool(self, v, t):
        """Convert the scalar to a TVM array."""
        return relay.const(v, type_to_np_dtype(t))

    def convert_nil(self, v, t):
        """Convert Nil to Relay."""
        return relay.Tuple([])

    def convert_dead(self, v, t):
        """Convert a dead value to Relay."""
        return dead_value(t)

    def convert_env(self, v, t):
        assert len(v) == 0
        return empty_env()

    def convert_tuple(self, v, t):
        return relay.Tuple([self(e, et) for e, et in
                            zip(v, t.elements)])

    def convert_tagged(self, v, t):  # pragma: no cover
        real_t = t.options.get(v.tag)
        ctr = get_union_ctr(v.tag, real_t)
        conv_val = self(v.value, real_t)
        return ctr(conv_val)

    def convert_type(self, v, t):
        return to_relay_type(v)


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

        graph, handles_cst, handles_params = return_handles(graph)

        self.module = relay.Module({})
        self.types = TypeHelper()
        self.types.initialize(self.module, mng)
        self.make_const = RelayConstantConverter(context)
        self.universe_helper = None

        # Analyze and create a global union type of all the possible types
        # and then use it for all union values.

        function_map = {}
        self.node_map = {}
        self.graph_map = {}

        for g in mng.graphs:
            if g.parent is None:
                if g is graph:
                    self.graph_map[g] = relay.GlobalVar("main")
                else:
                    self.graph_map[g] = relay.GlobalVar(g.debug.debug_name)

        for g in mng.graphs:
            if g.parent is None:
                function_map[self.graph_map[g]] = \
                    self.convert_func(g)

        self.types.finalize(self.module)
        add_functions(self.module, function_map)

        self.module = optimize(self.module)

        exec = relay.create_executor(mod=self.module, ctx=context,
                                     target=target)
        res = exec.evaluate(self.module["main"])

        res = handle_wrapper(res, handles_cst, handles_params)

        def f(*args):
            return wrap_result(res(*args))
        return f

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
        vname = "handle_seq" + graph.debug.debug_name
        i = 0
        for p in graph.parameters:
            self.node_map[p] = self.on_parameter(p)

        params = [self.ref(p) for p in graph.parameters]

        useq = []
        for node in toposort(graph.output, NodeVisitor()):
            if any(p.abstract.xtype() is UniverseType
                   for p in node.inputs[1:]):
                useq.append(node)
                self.node_map[node] = relay.var(f"{vname}.{i}")
                i += 1
            elif node.is_apply():
                self.node_map[node] = self.on_apply(node)
            elif node.is_constant_graph():
                self.node_map[node] = self.on_graph(node)
            elif node.is_constant():
                self.node_map[node] = self.on_constant(node)

        out = self.ref(graph.output)

        for uop in reversed(useq):
            assert node.is_apply()
            out = relay.Let(
                self.node_map[uop],
                self.on_apply(uop),
                out)

        res = relay.Function(params, out,
                             ret_type=to_relay_type(graph.output.abstract))

        return res


compiler = CompileGraph()


class DummyManager:
    all_nodes = []


class RelayInputConverter(Converter):
    """Convert values to Relay."""

    def __init__(self, context):
        """Set the context."""
        self.context = context
        self.cst_conv = RelayConstantConverter(self.context)
        mod = relay.Module({})
        th = TypeHelper()
        th.initialize(mod, DummyManager())
        th.finalize(mod)
        target = context.MASK2STR[context.device_type]
        if target == 'cpu':
            target = 'llvm'
        self.intrp = relay.create_executor(mod=mod, ctx=context, target=target)

    def convert_array(self, v, t):
        """Make a TVM array from a numpy array."""
        return interpreter.TensorValue(tvm.ndarray.array(v, self.context))

    def convert_scalar(self, v, t):
        """Convert the scalar to a TVM array."""
        return relay_from_scalar(v, type_to_np_dtype(t))

    def convert_bool(self, v, t):
        """Convert the scalar to a TVM array."""
        return relay_from_scalar(v, type_to_np_dtype(t))

    def convert_nil(self, v, t):
        """Convert Nil to Relay."""
        return interpreter.TupleValue()

    def convert_tuple(self, v, t):
        return interpreter.TupleValue(*[self(e, et) for e, et in
                                        zip(v, t.elements)])

    def convert_universe(self, v, t):
        return interpreter.TupleValue()

    def convert_handle(self, v, t):
        v = self(v.state, t.element)
        return interpreter.RefValue(v)

    def convert_tagged(self, v, t):
        cst = self.cst_conv.convert_tagged(v, t)
        return self.intrp.evaluate(cst)


class RelayOutputConverter(Converter):
    """Convert values from Relay."""

    def convert_array(self, v, t):
        """Make a numpy array from a TVM array."""
        return v.asnumpy()

    def convert_nil(self, v, t):
        """Convert Nil from relay."""
        assert len(v) == 0
        return None

    def convert_bool(self, v, t):
        """Convert the value to a boolean."""
        return v.asnumpy().item()

    def convert_scalar(self, v, t):
        """Convert the TVM array to a scalar."""
        return v.asnumpy().item()

    def convert_tuple(self, v, t):
        """Convert the value to a tuple."""
        return tuple(self(v, t)
                     for v, t in zip(v, t.elements))

    def convert_universe(self, v, t):
        """Convert a universe value."""
        return new_universe

    def convert_handle(self, v, t):
        return HandleInstance(self(v.value, t.element))

    def convert_tagged(self, v, t):
        tag = get_myia_tag(v.constructor)
        conv_val = self(v.fields[0], t.options.get(tag))
        return TaggedValue(tag, conv_val)


class RelayBackend(Backend):
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
        self.to_backend_value = RelayInputConverter(self.context)
        self.from_backend_value = RelayOutputConverter()

    def compile(self, graph, argspec, outspec):
        """Compiler a graph."""
        return self.compiler.run(graph, self.context, self.target)


def RelayBackendR(target, device_id):
    """Relay proxy."""
    return HandleBackend(RelayBackend(target, device_id))


__all__ = [
    'RelayBackend',
    'RelayBackendR',
]
