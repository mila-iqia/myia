"""Transforms a graph into lower-level code."""

from itertools import accumulate

import numpy as np
import tvm
from tvm import relay
from tvm.relay import adt
from tvm.relay.backend import interpreter

from ...abstract import AbstractTaggedUnion
from ...graph_utils import toposort
from ...ir import Graph, manage, sexp_to_node
from ...operations import Primitive, primitives as P
from ...operations.primitives import BackendPrimitive
from ...utils import HandleInstance, RandomStateWrapper, TaggedValue
from ...utils.variables import X, Y
from ...xtype import type_to_np_dtype, u32
from ..transform import convert_grad, get_prim_graph, return_handles
from . import Backend, Converter, relay_philox
from .relay_helpers import (
    TypeHelper,
    add_functions,
    dead_value,
    fill_reverse_tag_map,
    get_myia_tag,
    get_union_ctr,
    handle_wrapper,
    to_relay_type,
)

# Temporary primitive to replace make_handle in graphs
make_cell = BackendPrimitive(name="make_cell", defaults={})


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
    P.scalar_sin: relay.op.sin,
    P.scalar_cos: relay.op.cos,
    P.scalar_tan: relay.op.tan,
    P.scalar_trunc: relay.op.trunc,
    P.scalar_sign: relay.sign,
    P.scalar_abs: relay.abs,
    P.scalar_eq: relay.op.equal,
    P.scalar_lt: relay.op.less,
    P.scalar_gt: relay.op.greater,
    P.scalar_ne: relay.op.not_equal,
    P.scalar_le: relay.op.less_equal,
    P.scalar_ge: relay.op.greater_equal,
    P.bool_and: relay.op.logical_and,
    P.scalar_bit_and: relay.op.bitwise_and,
    P.scalar_bit_or: relay.op.bitwise_or,
    P.scalar_bit_xor: relay.op.bitwise_xor,
    P.scalar_bit_not: relay.op.bitwise_not,
    P.scalar_bit_lshift: relay.op.left_shift,
    P.scalar_bit_rshift: relay.op.right_shift,
    P.bool_or: relay.op.logical_or,
    P.bool_eq: relay.op.equal,
    P.bool_not: relay.op.logical_not,
    P.array_to_scalar: lambda x: x,
    P.dot: lambda x, y: relay.op.nn.dense(x, relay.op.transpose(y)),
    P.make_tuple: lambda *args: relay.Tuple(args),
    P.switch: relay.If,
    P.take: lambda w, i: relay.take(w, i, 0),
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
        res = relay.op.take(res, relay.const(0), mode="fast")
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
                "We currently support only full product on an array."
            )
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


def relay_tuple_setitem(c, t, idx, val):
    assert idx.is_constant(int)
    len_tuple = len(t.abstract.elements)
    tuple_value = c.ref(t)
    new_value = c.ref(val)
    value_idx = idx.value
    return relay.expr.Tuple(
        [relay.expr.TupleGetItem(tuple_value, i) for i in range(value_idx)]
        + [new_value]
        + [
            relay.expr.TupleGetItem(tuple_value, i)
            for i in range(value_idx + 1, len_tuple)
        ]
    )


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
    t_clause = adt.Clause(
        adt.PatternConstructor(rtag, [adt.PatternWildcard()]), relay.const(True)
    )
    f_clause = adt.Clause(adt.PatternWildcard(), relay.const(False))
    return adt.Match(c.ref(x), [t_clause, f_clause])


def relay_tagged(c, x, tag):
    """Implementation of tagged for Relay."""
    assert tag.is_constant(int)
    rtag = get_union_ctr(tag.value, None)
    return rtag(c.ref(x))


def relay_env_setitem(c, env, key, x):
    """Implementation of env_setitem for Relay."""
    return c.types.do_env_update(c.ref(env), key.value, c.ref(x))


def relay_env_getitem(c, env, key, dft):
    """Implementation of env_getitem for Relay."""
    return c.types.do_env_find(c.ref(env), key.value, c.ref(dft))


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
    return relay.op.transform.strided_slice(
        c.ref(a), start.value, stop.value, strides.value
    )


def relay_array_setitem(c, array, start, stop, strides, value):
    assert start.is_constant(tuple)
    assert stop.is_constant(tuple)
    assert strides.is_constant(tuple)

    v_start = relay.const(list(start.value))
    v_stop = relay.const(list(stop.value))
    v_strides = relay.const(list(strides.value))

    return relay.op.transform.strided_set(
        c.ref(array), c.ref(value), v_start, v_stop, v_strides
    )


def relay_argmax(c, v, dims):
    """Implementation of argmax for Relay."""
    v = c.ref(v)
    assert dims.is_constant(tuple)
    return relay.cast(relay.argmax(v, axis=dims.value, keepdims=True), "int64")


def relay_max_pool2d(c, img, psize, stride, pad, dil, ceil_mode):
    assert psize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert ceil_mode.is_constant(bool)
    assert dil.value == (1, 1)

    return relay.nn.max_pool2d(
        c.ref(img),
        psize.value,
        stride.value,
        pad.value,
        ceil_mode=ceil_mode.value,
    )


def relay_max_pool2d_grad(c, img, psize, stride, pad, dil, ceil_mode, dout):
    assert psize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert ceil_mode.is_constant(bool)
    assert dil.value == (1, 1)

    return relay.nn.max_pool2d_grad(
        c.ref(dout),
        c.ref(img),
        psize.value,
        stride.value,
        pad.value,
        ceil_mode=ceil_mode.value,
    )


def relay_array_max(c, a, dim):
    assert dim.is_constant(tuple)
    return relay.max(c.ref(a), axis=dim.value)


def relay_conv2d(c, img, w, stride, pad, dil, groups):
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert groups.is_constant(int)

    return relay.nn.conv2d(
        c.ref(img),
        c.ref(w),
        strides=stride.value,
        padding=pad.value,
        dilation=dil.value,
        groups=groups.value,
    )


def relay_conv2d_weight_grad(c, data, wsize, dout, stride, pad, dil, groups):
    # This implementation should match the one in pytorch backend
    # (myia.compile.backends.pytorch_conv_grad.conv2d_weight)

    assert wsize.is_constant(tuple)
    assert stride.is_constant(tuple)
    assert pad.is_constant(tuple)
    assert dil.is_constant(tuple)
    assert groups.is_constant(int)

    batch, in_channel, in_h, in_w = data.abstract.xshape()
    out_channel, _, filter_h, filter_w = wsize.value
    grad_sh0, grad_sh1, grad_h, grad_w = dout.abstract.xshape()
    pad_h, pad_w = pad.value

    data = c.ref(data)
    dout = c.ref(dout)

    fpad_h = pad_h * 2
    fpad_w = pad_w * 2
    fpad_top = (pad_h + 1) // 2
    fpad_left = (pad_w + 1) // 2
    fpad_bottom = fpad_h - fpad_top
    fpad_right = fpad_w - fpad_left

    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride.value[0] - 1 + fpad_top + fpad_bottom
    ) // dil.value[0] + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride.value[1] - 1 + fpad_left + fpad_right
    ) // dil.value[1] + 1

    dout = relay.tile(dout, [1, in_channel // groups.value, 1, 1])
    dout = relay.reshape(dout, [-1, 1, 0, 0])
    data = relay.reshape(data, [1, -1, 0, 0])

    d = relay.nn.conv2d(
        data,
        dout,
        strides=dil.value,
        padding=pad.value,
        dilation=stride.value,
        groups=batch * in_channel,
    )

    conv_sh1 = grad_sh0 * grad_sh1 * (in_channel // groups.value)
    d = relay.reshape(
        d,
        [batch, conv_sh1 // batch, padded_weight_grad_h, padded_weight_grad_w],
    )
    d = relay.sum(d, axis=0)

    if groups.value > 1:
        d = relay.reshape(
            d,
            [
                grad_sh1,
                in_channel // groups.value,
                padded_weight_grad_h,
                padded_weight_grad_w,
            ],
        )
    else:
        d = relay.reshape(
            d,
            [
                in_channel // groups.value,
                grad_sh1,
                padded_weight_grad_h,
                padded_weight_grad_w,
            ],
        )
        d = relay.transpose(d, [1, 0, 2, 3])

    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        d = relay.strided_slice(
            d, begin=[0, 0, 0, 0], end=[None, None, filter_h, filter_w]
        )
    return d


def relay_conv_transpose2d(
    c, input, weight, stride, padding, output_padding, groups, dilation
):
    """Implement conv2d_transpose using 10 relay calls including conv2d.

    Support all values for groups, dilation, strides, padding and
    output padding.
    Based on Theano implementation (2020/04/14):
    https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/abstract_conv.py#L2927
    Need implementation of operation relay.nn.dilate
    in TVM relay backend
    """

    assert stride.is_constant(tuple)
    assert padding.is_constant(tuple)
    assert output_padding.is_constant(tuple)
    assert dilation.is_constant(tuple)
    assert groups.is_constant(int)

    data_shape = input.abstract.xshape()
    kern_shape = weight.abstract.xshape()
    n, _, h_in, w_in = data_shape
    filter_h, filter_w = kern_shape[2:]
    strides = stride.value
    padding = padding.value
    dilation = dilation.value
    output_padding = output_padding.value
    groups = groups.value
    data = c.ref(input)
    weight = c.ref(weight)

    h_out = (
        (h_in - 1) * strides[0]
        - 2 * padding[0]
        + dilation[0] * (filter_h - 1)
        + output_padding[0]
        + 1
    )
    w_out = (
        (w_in - 1) * strides[1]
        - 2 * padding[1]
        + dilation[1] * (filter_w - 1)
        + output_padding[1]
        + 1
    )

    data_dilated = relay.nn.dilate(data, (1, 1) + strides)
    data_padded = relay.nn.pad(
        data_dilated,
        ((0, 0), (0, 0), (0, output_padding[0]), (0, output_padding[1]),),
    )

    # Pre-process kernel,
    # from (m0, m1, m2, m3) to (m1 * g, m0 // g, m2, m3).
    mshp0 = kern_shape[0] // groups
    c_out = kern_shape[1] * groups
    kern = relay.reshape(weight, (groups, mshp0) + kern_shape[1:])
    # => (g, m0 // g, m1, m2, m3)
    kern = relay.op.transpose(kern, axes=(1, 0, 2, 3, 4))
    # => (m0 // g, g, m1, m2, m3)
    kern = relay.reshape(kern, (mshp0, c_out, kern_shape[-2], kern_shape[-1]))
    # => (m0 // g, m1 * g, m2, m3)
    kern = relay.op.transpose(kern, (1, 0, 2, 3))
    # => (m1 * g, m0 // g, m2, m3)
    # Kernel 2 latest dimensions must be flipped
    kern = relay.op.transform.reverse(kern, 2)
    kern = relay.op.transform.reverse(kern, 3)
    # End pre-processing kernel.

    img = relay.nn.conv2d(
        data_padded,
        kern,
        groups=groups,
        channels=c_out,
        padding=[(kern_shape[2 + i] - 1) * dilation[i] for i in range(2)],
        dilation=dilation,
    )

    if any(p != 0 for p in padding):
        img = relay.op.transform.strided_slice(
            data=img,
            begin=[0, 0, padding[0], padding[1]],
            end=[n + 1, c_out + 1, h_out + padding[0], w_out + padding[1]],
        )

    return img


def relay_concat(c, x, dim):
    assert dim.is_constant(int)

    xr = c.ref(x)
    inputs = [
        relay.expr.TupleGetItem(xr, i) for i in range(len(x.abstract.elements))
    ]
    return relay.concatenate(inputs, dim.value)


def relay_split(c, x, sections, dim):
    assert sections.is_constant(tuple)
    assert dim.is_constant(int)

    sections = tuple(accumulate(sections.value))[:-1]
    return relay.split(c.ref(x), sections, dim.value).astuple()


def relay_gather(c, data, axis, indices):
    assert axis.is_constant(int)
    return relay.gather(c.ref(data), axis.value, c.ref(indices))


def relay_scatter(c, inp, dim, index, src):
    assert dim.is_constant(int)
    return relay.scatter(c.ref(inp), c.ref(index), c.ref(src), dim.value)


def relay_make_cell(c, v, u):
    return relay.Tuple((c.ref(u), relay.expr.RefCreate(c.ref(v))))


# Proper sequencing is handled in convert_func() below
def relay_universe_setitem(c, u, h, v):
    return relay.RefWrite(c.ref(h), c.ref(v))


def relay_universe_getitem(c, u, h):
    return relay.RefRead(c.ref(h))


def relay_take_grad_inp(c, _nb_indices, _indices, _values):
    assert _nb_indices.is_constant(int)
    values = c.ref(_values)
    r_indices = relay.reshape(
        c.ref(_indices), tuple(_indices.abstract.xshape()) + (1,)
    )
    n_rows = _nb_indices.value
    n_cols = _values.abstract.xshape()[-1]
    outputs = []
    indices_dtype = type_to_np_dtype(_indices.abstract.element.xtype())
    out_dtype = type_to_np_dtype(_values.abstract.element.xtype())
    for i in range(n_rows):
        select_entries = relay.equal(r_indices, relay.const(i, indices_dtype))
        casted_select = relay.cast(select_entries, out_dtype)
        select_dout = relay.multiply(casted_select, values)
        reshape_out = relay.reshape(select_dout, (-1, n_cols))
        vector = relay.sum(reshape_out, 0)
        outputs.append(relay.reshape(vector, (1, n_cols)))
    return relay.concatenate(outputs, 0)


def relay_random_initialize(c, ref_seed):
    """Create a random state for Philox2x32 RNG.

    State is a couple (key, counter).
    key is given seed, or a default value if seed is None.
    counter starts with 0 and is incremented after each generation batch.
    """
    assert ref_seed.is_constant(type(None)) or ref_seed.is_constant(int)
    seed = ref_seed.value
    key = relay.const(seed, "uint32")
    counter = relay.const(0, "uint32")
    rstate = relay.Tuple((key, counter))
    return rstate


def relay_random_uint32(c, ref_rstate, ref_shape):
    """Generate a random tensor using Philox2x32 RNG."""
    assert ref_shape.is_constant(tuple)
    shape = ref_shape.value
    relay_state = c.ref(ref_rstate)
    # Compute output size.
    output_size = 1
    for dim in shape:
        output_size *= dim
    # Generate random uint32 values.
    key = relay.TupleGetItem(relay_state, 0)
    counter = relay.TupleGetItem(relay_state, 1)
    impl = relay_philox.Philox2x32(output_size)
    ctr = impl.generate_relay_counter_array(counter)
    random = impl.philox_2x(ctr, key)
    # Reshape vector to expected shape.
    if shape:
        # Reshape vector to output shape.
        random = relay.op.reshape(random, shape)
    else:
        # Convert 1-element vector to scalar
        random = relay.op.take(random, relay.const(0), mode="fast")
    # Generate next state: same key, counter + 1
    next_rstate = relay.Tuple(
        (key, relay.add(counter, relay.const(1, "uint32")))
    )
    # Return next state and random tensor.
    return relay.Tuple((next_rstate, random))


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
    P.tuple_setitem: relay_tuple_setitem,
    P.casttag: relay_casttag,
    P.hastag: relay_hastag,
    P.tagged: relay_tagged,
    P.env_setitem: relay_env_setitem,
    P.env_getitem: relay_env_getitem,
    P.unsafe_static_cast: relay_unsafe_static_cast,
    P.array_getitem: relay_array_getitem,
    P.array_setitem: relay_array_setitem,
    P.argmax: relay_argmax,
    P.max_pool2d: relay_max_pool2d,
    P.max_pool2d_grad: relay_max_pool2d_grad,
    P.array_max: relay_array_max,
    P.conv2d: relay_conv2d,
    P.conv2d_weight_grad: relay_conv2d_weight_grad,
    P.conv_transpose2d: relay_conv_transpose2d,
    P.concat: relay_concat,
    P.split: relay_split,
    P.universe_setitem: relay_universe_setitem,
    P.universe_getitem: relay_universe_getitem,
    P.take_grad_inp: relay_take_grad_inp,
    P.random_initialize: relay_random_initialize,
    P.random_uint32: relay_random_uint32,
    make_cell: relay_make_cell,
    P.gather: relay_gather,
    P.scatter: relay_scatter,
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
            self.register(k, lambda c, *args, v=v: v(*[c.ref(a) for a in args]))

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

    def _visit_array_cast(self, node):
        return [node.inputs[1]]

    def _visit_scalar_to_array(self, node):
        return [node.inputs[1]]

    def _visit_unsafe_static_cast(self, node):
        return [node.inputs[1]]

    def _visit_scalar_cast(self, node):
        return [node.inputs[1]]

    def __call__(self, node):
        """Don't visit called primitives."""
        if node.inputs:
            fn = node.inputs[0]
            if fn.is_constant(Primitive):
                prim = fn.value
                visit = getattr(self, f"_visit_{prim}", None)
                if visit is None:
                    return node.inputs[1:]
                return visit(node)
            else:
                return node.inputs
        elif node.is_constant_graph():
            return [
                fv
                if not isinstance(fv, Graph)
                else list(fv.manager.graph_constants[fv])[0]
                for fv in node.value.free_variables_total
            ]
        return []


def in_graph(g):
    def filter(node):
        if node.graph is None:
            return "follow"
        elif node.graph is g:
            return "follow"
        else:
            return "exclude"

    return filter


class RelayConstantConverter(Converter):
    """Convert values to Relay constants."""

    def __init__(self, context, types):
        """Set the context."""
        self.context = context
        self.types = types

    def convert_array(self, v, t):  # pragma: no cover
        """Make a TVM array from a numpy array."""
        return relay.const(tvm.runtime.ndarray.array(v, self.context))

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
        return self.types.build_default_env_val()

    def convert_handle(self, v, t):
        return relay.expr.RefCreate(self(v.state, v.abstract or t.element))

    def convert_tuple(self, v, t):
        return relay.Tuple([self(e, et) for e, et in zip(v, t.elements)])

    def convert_tagged(self, v, t):
        real_t = t.options.get(v.tag)
        ctr = get_union_ctr(v.tag, real_t)
        conv_val = self(v.value, real_t)
        return ctr(conv_val)


class CompileGraph:
    """Step to convert a myia graph to a relay graph.

    Inputs:
        graph: A graph

    Outputs:
        output: a wrapped relay graph
    """

    def run(self, graph, context, target, exec_kind):
        """Convert the graph into a relay callable."""
        mng = manage(graph)

        graph, handles_params = return_handles(graph)

        mng.keep_roots(graph)

        self.module = tvm.IRModule({})
        self.types = TypeHelper()
        self.types.initialize(self.module, mng)
        self.make_const = RelayConstantConverter(context, self.types)
        self.universe_helper = None
        self.i = 0

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
                    # Mangle user names
                    name = "_" + g.debug.debug_name
                    self.graph_map[g] = relay.GlobalVar(name)

        for g in self.graph_map.keys():
            function_map[self.graph_map[g]] = self.convert_func(g)

        add_functions(self.module, function_map)

        vm = relay.create_executor(
            mod=self.module, ctx=context, target=target, kind=exec_kind
        )
        res = vm.evaluate()

        fill_reverse_tag_map()

        res = handle_wrapper(res, handles_params)

        return res

    def on_parameter(self, node):
        """Convert a parameter node."""
        return relay.var(
            node.debug.debug_name, type_annotation=to_relay_type(node.abstract)
        )

    def on_apply(self, node):
        """Convert an Apply node."""
        if node.inputs[0].is_constant(Primitive):
            fn = node.inputs[0].value
            conv = MAP.get(fn)
            if conv is not None:
                return conv(self, *node.inputs[1:])
        return relay.Call(
            self.ref(node.inputs[0]), [self.ref(i) for i in node.inputs[1:]]
        )

    def on_constant(self, node):
        """Convert a constant node."""
        if node.is_constant(Primitive):
            return self.convert_func(
                get_prim_graph({}, node.value, node.abstract)
            )
        return self.make_const(node.value, node.abstract)

    def ref(self, node):
        """Return the value for a node."""
        return self.node_map[node]

    def convert_func(self, graph):
        """Convert a graph."""
        for p in graph.parameters:
            self.node_map[p] = self.on_parameter(p)

        params = [self.ref(p) for p in graph.parameters]

        seq = []
        for node in toposort(graph.output, NodeVisitor(), in_graph(graph)):
            if node in self.node_map:
                continue
            elif node.is_constant_graph() and node.value.parent is None:
                self.node_map[node] = self.graph_map[node.value]
            else:
                self.node_map[node] = relay.var(f"seq.{self.i}")
                self.i += 1
                seq.append(node)

        out = self.ref(graph.output)

        for op in reversed(seq):
            var = self.node_map[op]
            if op.is_apply():
                val = self.on_apply(op)
            elif op.is_constant_graph():
                val = self.convert_func(op.value)
            elif op.is_constant():
                val = self.on_constant(op)
                # This forces the rebuild of constants every time they
                # are encountered since they may be shared amongst
                # multiple graphs and it causes problems otherwise.
                del self.node_map[op]
            else:
                raise AssertionError(f"Bad node for sequence: {op}")
            out = relay.Let(var, val, out)

        return relay.Function(
            params, out, ret_type=to_relay_type(graph.output.abstract)
        )


compiler = CompileGraph()


class RelayInputConverter(Converter):
    """Convert values to Relay."""

    def __init__(self, context, exec_kind):
        """Set the context."""
        self.context = context
        self.exec_kind = exec_kind
        self.th = TypeHelper()
        self.cst_conv = RelayConstantConverter(self.context, self.th)

    def convert_array(self, v, t):
        """Make a TVM array from a numpy array."""
        return tvm.runtime.ndarray.array(v, self.context)

    def convert_scalar(self, v, t):
        """Convert the scalar to a TVM array."""
        return tvm.runtime.ndarray.array(
            getattr(np, type_to_np_dtype(t))(v), self.context
        )

    def convert_bool(self, v, t):
        """Convert the scalar to a TVM array."""
        return tvm.runtime.ndarray.array(np.bool_(v), self.context)

    def convert_nil(self, v, t):
        """Convert Nil to Relay."""
        return ()

    def convert_tuple(self, v, t):
        return tuple(self(e, et) for e, et in zip(v, t.elements))

    def convert_random_state(self, v, t):
        return tuple(self.convert_scalar(e, u32) for e in v.state)

    def convert_universe(self, v, t):
        return ()

    def convert_handle(self, v, t):
        v = self(v.state, t.element)
        return interpreter.RefValue(v)

    def convert_tagged(self, v, t):
        mod = tvm.IRModule({})
        self.th.initialize(mod, None)
        cst = self.cst_conv.convert_tagged(v, t)
        mod["main"] = relay.Function([], cst)
        vm = relay.create_executor(
            ctx=self.context, mod=mod, kind=self.exec_kind
        )
        return vm.evaluate()()

    def convert_type(self, v, t):
        # abstract type will be replaced with an integer type as placeholder
        # (see to_relay_type(AbstractType), so we must return an integer
        # of same type here.
        return np.int32(0)


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
        return tuple(self(v, t) for v, t in zip(v, t.elements))

    def convert_handle(self, v, t):
        return HandleInstance(self(v.value, t.element))

    def convert_tagged(self, v, t):
        tag = get_myia_tag(v.tag)
        try:
            conv_val = self(v[0], t.options.get(tag))
        except TypeError:
            conv_val = self(v.fields[0], t.options.get(tag))
        return TaggedValue(tag, conv_val)

    def convert_random_state(self, v, t):
        return RandomStateWrapper(tuple(el.asnumpy().item() for el in v))


def make_handle_to_make_cell(g):
    """Replace uset(*make_handle(typ), value) by make_cell(value, U).

    This is because RefCreate both creates the reference and sets it.
    """
    mng = manage(g)
    for node in list(mng.all_nodes):
        equiv = node.match(
            (
                P.universe_setitem,
                (P.tuple_getitem, X, 0),
                (P.tuple_getitem, X, 1),
                Y,
            )
        )
        if equiv:
            x = equiv[X]
            if x.is_apply(P.make_handle):
                new_handle_node = sexp_to_node(
                    (make_cell, equiv[Y], x.inputs[2]), node.graph
                )
                mng.replace(x, new_handle_node)
                mng.replace(node, node.inputs[1])


class RelayBackend(Backend):
    """Backend based on Relay.

    Backend options:

        :target: the target device class ('cpu', 'cuda', ...)
        :device_id: the target device identifier (an int)
        :exec_kind: a string ('vm' or 'debug')

    """

    def __init__(self, target, device_id, exec_kind):
        """Create a Relay backend for the given device."""
        device_id = int(device_id)
        self.context = tvm.runtime.ndarray.context(target, device_id)
        if target == "cpu":
            target = "llvm"
        self.target = target
        if not self.context.exist:
            raise RuntimeError(
                "No hardware to support selected target "
                f"'{target}' on device {device_id}"
            )
        if exec_kind not in ("vm", "debug"):
            raise ValueError(f"Invalid exec_kind: {exec_kind}")
        self.exec_kind = exec_kind
        self.compiler = compiler
        self.to_backend_value = RelayInputConverter(
            self.context, self.exec_kind
        )
        self.from_backend_value = RelayOutputConverter()

    def compile(self, graph, argspec, outspec):
        """Compiler a graph."""
        make_handle_to_make_cell(graph)
        graph = convert_grad(graph)
        return self.compiler.run(
            graph, self.context, self.target, self.exec_kind
        )


__all__ = ["RelayBackend"]
