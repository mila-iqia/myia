"""Linear implementation using NNVM."""

from itertools import count

import nnvm.compiler
import nnvm.symbol as sym
import numpy as np
import tvm
from nnvm.compiler import graph_attr
from tvm.contrib import graph_runtime

from ...abstract import AbstractArray
from ...dtype import Nil, type_to_np_dtype
from ...prim import Primitive, ops as P
from ..transform import CompileGraphs, nonlinear_ops
from ..utils import get_outputs
from . import Backend

SIMPLE_MAP = {
    P.scalar_add: sym.elemwise_add,
    P.scalar_sub: sym.elemwise_sub,
    P.scalar_mul: sym.elemwise_mul,
    P.scalar_div: sym.elemwise_div,
    P.scalar_mod: sym.elemwise_mod,
    P.scalar_pow: sym.elemwise_pow,
    P.scalar_floor: sym.floor,
    P.scalar_max: sym.broadcast_max,
    P.scalar_uadd: lambda x: x,
    P.scalar_usub: sym.negative,
    P.scalar_exp: sym.exp,
    P.scalar_log: sym.log,
    P.scalar_tanh: sym.tanh,

    P.scalar_eq: sym.broadcast_equal,
    P.scalar_lt: sym.broadcast_less,
    P.scalar_gt: sym.broadcast_greater,
    P.scalar_ne: sym.broadcast_not_equal,
    P.scalar_le: sym.broadcast_less_equal,
    P.scalar_ge: sym.broadcast_greater_equal,
    # P.bool_and: sym.logical_and,
    # P.bool_or: sym.logical_or
    P.bool_eq: sym.broadcast_equal,

    P.switch: sym.where,

    P.array_to_scalar: lambda x: x
}


def nnvm_bool_not(c, arg):
    """Implementation of boolean not."""
    t = arg.abstract.dtype()
    zero = c.make_constant(0, nnvm_type=type_to_np_dtype(t))
    return sym.broadcast_equal(zero, c.ref(arg))


def nnvm_distribute(c, v, shp):
    """Implementation of distribute."""
    nv = c.ref(v)
    assert shp.is_constant(tuple)
    if shp.value == ():
        shp = (1,)
    else:
        shp = shp.value
    vshp = ashape(v)
    if len(shp) != len(vshp):
        # We need to pad the shape
        nv = sym.expand_dims(nv, axis=0, num_newaxis=len(shp) - len(vshp))
    if shp == vshp:
        return nv
    return sym.broadcast_to(nv, shape=shp)


def nnvm_dot(c, a, b):
    """Implementation of dot."""
    na = c.ref(a)
    nb = c.ref(b)
    return sym.dense(na, sym.transpose(nb, axes=(1, 0)), units=b.shape[1],
                     use_bias=False)


def nnvm_array_map(c, fn, *array):
    """Implementation of array_map."""
    assert fn.is_constant(Primitive)
    fn = fn.value
    return SIMPLE_MAP[fn](*[c.ref(a) for a in array])


def nnvm_array_reduce(c, fn, array, shape):
    """Implementation of array_reduce."""
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
        axis = list(i for i, t in enumerate(ts) if t == 1)
        if len(axis) == 1:
            axis = axis[0]
        res = sym.sum(ary, axis=axis, keepdims=1)
        if len(tshp) < len(ashp):
            if tshp == ():
                tshp = (1,)
            res = sym.reshape(res, shape=tshp)
        return res
    else:
        raise NotImplementedError(f"reduce with {fn}")


def nnvm_transpose(c, a, ax):
    """Implementation of transpose."""
    na = c.ref(a)
    assert ax.is_constant(tuple)
    return sym.transpose(na, axes=ax.value)


def nnvm_reshape(c, v, shp):
    """Implementation of reshape."""
    nv = c.ref(v)
    assert shp.is_constant(tuple)
    if shp.value == ():
        shp = (1,)
    else:
        shp = shp.value
    return sym.reshape(nv, shape=shp)


COMPLEX_MAP = {
    P.bool_not: nnvm_bool_not,
    P.distribute: nnvm_distribute,
    P.dot: nnvm_dot,
    P.array_map: nnvm_array_map,
    P.array_reduce: nnvm_array_reduce,
    P.transpose: nnvm_transpose,
    P.scalar_to_array: lambda c, x, t: c.ref(x),
    P.reshape: nnvm_reshape,
}


def nnvm_val(val, dtype, ctx):
    """Build an NNVM value on-device."""
    vv = np.array(val, dtype=dtype,
                  copy=False, ndmin=1)
    v = tvm.ndarray.empty(shape=vv.shape, dtype=vv.dtype,
                          ctx=ctx)
    v.copyfrom(vv)
    return v


class NNVMRunner:
    """Adapter to run an NNVM module."""

    def __init__(self, mod, input_names, input_types, output_specs, context):
        """Intialize the runner.

        Arguments:
            mod: NNVM compiled module
            input_names: list of the names of inputs (in order)
            output_specs: list of shape and dtype for outputs
                          [(shp0, dtype0), ...]
            context: TVMContext for the runtime and arrays

        """
        self.mod = mod
        self.input_names = input_names
        self.input_types = input_types
        self.output_specs = output_specs
        self.context = context

    def __call__(self, *args):
        """Run the module on the arguments."""
        assert len(args) == len(self.input_names)
        nnvm_args = dict()
        for n, tp, v in zip(self.input_names, self.input_types, args):
            nnvm_args[n] = v
        self.mod.set_input(**nnvm_args)
        self.mod.run()
        outs = [tvm.nd.empty(spec[0], dtype=spec[1], ctx=self.context)
                for spec in self.output_specs]
        for i, out in enumerate(outs):
            out = self.mod.get_output(i, out)
        return outs


def ashape(a):
    """Get an array shape.

    Handles NNVM brain-damage around empty shapes.
    """
    shp = a.shape
    if shp == ():
        return (1,)
    return shp


class NNVMConverter:
    """Convert a linear portion of the graph to an NNVM function."""

    def __init__(self, simple_map=None, complex_map=None):
        """Create a converter."""
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
            self.register(k, lambda c, *args, v=v: v(*[self.ref(a)
                                                       for a in args]))

    def register_complex(self, map):
        """Register complex conversions."""
        for k, v in map.items():
            self.register(k, v)

    def make_constant(self, val, nnvm_type):
        """Make a utility constant that is not part of the graph."""
        key = (val, nnvm_type)
        if key not in self.constant_vars:
            name = f"_cst{val}{nnvm_type}"
            self.constants[name] = nnvm_val([val], dtype=nnvm_type,
                                            ctx=self.context)
            self.constant_vars[key] = sym.Variable(name)
            self.types[name] = nnvm_type
            self.shapes[name] = (1,)
        return self.constant_vars[key]

    def ref(self, n):
        """Resolve a reference to a node."""
        def setn(name, n):
            """Associate name with n."""
            self.eqv[n] = sym.Variable(name)
            if isinstance(t, AbstractArray):
                te = t.element.dtype()
                self.types[name] = type_to_np_dtype(te)
                self.shapes[name] = ashape(n)
            elif n.is_constant_graph():  # pragma: no cover
                raise Exception("This isn't tested")
                self.types[name] = 'int64'
                self.shapes[name] = (1,)
            else:
                te = t.dtype()
                self.types[name] = type_to_np_dtype(te)
                self.shapes[name] = (1,)

        t = n.abstract
        if n.is_constant() and not n.is_constant_graph():
            name = f"cst{next(self.c)}"
            te = t.dtype()
            self.constants[name] = nnvm_val([n.value],
                                            dtype=type_to_np_dtype(te),
                                            ctx=self.context)

            setn(name, n)
        elif n not in self.eqv:
            name = f"i{next(self.c)}"
            self.inputs.append(n)
            self.input_names.append(name)
            setn(name, n)
        return self.eqv[n]

    def convert(self, lst, context):
        """Converts the list of nodes to a runnable form.

        All the nodes in the list must represent linear flow (no calls,
        branches, ...)

        Returns:
            (fn, inputs, outputs):

            - fn: A callable function
            - inputs: the list of inputs nodes whose values should be
                      provided to the function
            - outputs: the list of output nodes corresponding to the
                       outputs of the function

        Notes:
            This implementation converts the nodes to NNVM and compiles it.

        """
        self.c = count()
        self.eqv = {}
        self.inputs = []
        self.input_names = []
        self.constants = {}
        self.constant_vars = {}
        self.shapes = {}
        self.types = {}
        self.context = context

        for n in lst:
            assert n.is_apply()
            assert n.inputs[0].is_constant(Primitive)
            fn = n.inputs[0].value
            conv = self.mapping.get(fn, None)
            if conv is not None:
                self.eqv[n] = conv(self, *n.inputs[1:])
            else:
                raise NotImplementedError(fn)

        outputs = get_outputs(lst, lst[0].graph.manager.uses,
                              set(self.eqv.keys()))

        inmap = dict((self.eqv[i], i) for i in self.inputs)

        # Check for empty functions
        if all(self.eqv[o] in inmap for o in outputs):
            return None, [inmap[self.eqv[o]] for o in outputs], outputs

        target = context.MASK2STR[context.device_type]
        if target == 'cpu':
            nnvm_target = 'llvm'
        elif target == 'gpu':
            nnvm_target = 'cuda -libs=cublas'

        g = nnvm.graph.create(sym.Group(list(self.eqv[o] for o in outputs)))
        dg, lib, params = nnvm.compiler.build(
            g, target=nnvm_target, shape=self.shapes, dtype=self.types,
            params=self.constants)

        shape = dg.json_attr('shape')
        types = dg.json_attr('dtype')
        index = dg.index

        def spec(entry_id):
            return (shape[entry_id],
                    graph_attr.TCODE_TO_DTYPE[types[entry_id]])

        output_specs = [spec(index.entry_id(x)) for x in index.output_entries]
        assert len(output_specs) == len(outputs)

        module = graph_runtime.create(dg, lib, self.context)

        for n, p in params.items():
            module.set_input(n, p)

        input_types = [self.types[i] for i in self.input_names]
        return (NNVMRunner(module, self.input_names,
                           input_types, output_specs, self.context),
                self.inputs, outputs)


converter = NNVMConverter(simple_map=SIMPLE_MAP, complex_map=COMPLEX_MAP)
nnvm_convert = converter.convert


class NNVMBackend(Backend):
    """Backend to compile for NNVM.

    Backend options:
        target: the target device class ('cpu', 'cuda')
        device_id: the target device identifier (an int)

    """

    def __init__(self, target='cpu', device_id=0):
        """Create a NNVM backend for the given device."""
        device_id = int(device_id)
        self.context = tvm.ndarray.context(target, device_id)
        if not self.context.exist:
            raise RuntimeError("No hardware to support selected target/device")
        self.compiler = CompileGraphs(
            lambda l: converter.convert(l, context=self.context),
            nonlinear_ops, self)

    def compile(self, graph, *others):
        """Compile a graph."""
        return self.compiler.compile_and_link(graph)

    def to_numpy(self, v):
        """Make a numpy array from a NNVM array."""
        return v.asnumpy()

    def from_numpy(self, a):
        """Make an NNVM array from a numpy array."""
        return tvm.ndarray.array(a, self.context)

    def to_scalar(self, v):
        """Convert the NNVM array to a scalar."""
        if v is None:
            return v
        else:
            return v.asnumpy().item()

    def from_scalar(self, s, t):
        """Convert the scalar to an NNVM array."""
        if t == Nil:
            return None
        dt = type_to_np_dtype(t)
        return self.from_numpy(np.array(s, dtype=dt, copy=False, ndmin=1))
