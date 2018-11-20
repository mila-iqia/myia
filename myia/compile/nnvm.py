"""Linear implementation using NNVM."""

import numpy as np
from itertools import count

import nnvm.compiler
import nnvm.symbol as sym
import tvm
from nnvm.compiler import graph_attr
from tvm.contrib import graph_runtime

from .utils import get_outputs

from ..dtype import type_to_np_dtype, ismyiatype, Array
from ..prim import Primitive, ops as P


SIMPLE_MAP = {
    P.scalar_add: sym.elemwise_add,
    P.scalar_sub: sym.elemwise_sub,
    P.scalar_mul: sym.elemwise_mul,
    P.scalar_div: sym.elemwise_div,
    P.scalar_mod: sym.elemwise_mod,
    P.scalar_pow: sym.elemwise_pow,
    P.scalar_floor: sym.floor,
    P.scalar_uadd: lambda x: x,
    P.scalar_usub: sym.negative,
    P.scalar_exp: sym.exp,
    P.scalar_log: sym.log,
    # This is not right tangent vs hyperbolic tangent
    # P.scalar_tan: sym.tanh,

    P.scalar_eq: sym.broadcast_equal,
    P.scalar_lt: sym.broadcast_less,
    P.scalar_gt: sym.broadcast_greater,
    P.scalar_ne: sym.broadcast_not_equal,
    P.scalar_le: sym.broadcast_less_equal,
    P.scalar_ge: sym.broadcast_greater_equal,
    # P.bool_and: sym.logical_and,
    # P.bool_or: sym.logical_or
    P.bool_eq: sym.broadcast_equal,

    P.scalar_to_array: lambda x: x
}


def nnvm_bool_not(c, arg):
    """Implementation of boolean not."""
    zero = c.make_constant(0, nnvm_type=nnvm_type_map(arg.type))
    return sym.broadcast_equal(zero, c.ref(arg))


def nnvm_distribute(c, v, shp):
    """Implementation of distribute."""
    nv = c.ref(v)
    assert shp.is_constant()
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
        axis = list(i for i, t in enumerate(tshp) if t == 1)
        if len(axis) == 1:
            axis = axis[0]
        res = sym.sum(ary, axis=axis, keepdims=1)
        if len(tshp) < len(ashp):
            axis = tuple(range(len(tshp), len(ashp)))
            res = sym.sum(res, axis=axis)
        return res
    else:
        raise NotImplementedError(f"reduce with {fn}")


def nnvm_transpose(c, a, ax):
    """Implementation of transpose."""
    na = c.ref(a)
    assert ax.is_constant(tuple)
    return sym.transpose(na, axes=ax.value)


COMPLEX_MAP = {
    P.bool_not: nnvm_bool_not,
    P.distribute: nnvm_distribute,
    P.dot: nnvm_dot,
    P.array_map: nnvm_array_map,
    P.array_reduce: nnvm_array_reduce,
    P.transpose: nnvm_transpose,
}


def nnvm_type_map(type):
    """Map a numpy type to an NNVM type."""
    dt = type_to_np_dtype(type)
    if dt == 'bool':
        dt = 'uint8'
    return dt


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
        self._outs = [tvm.nd.empty(spec[0], dtype=spec[1], ctx=context)
                      for spec in self.output_specs]

    def __call__(self, *args):
        """Run the module on the arguments."""
        assert len(args) == len(self.input_names)
        nnvm_args = dict()
        for n, tp, v in zip(self.input_names, self.input_types, args):
            nnvm_args[n] = np.array(v, dtype=tp, copy=False, ndmin=1)
        self.mod.set_input(**nnvm_args)
        self.mod.run()
        for i, out in enumerate(self._outs):
            out = self.mod.get_output(i, out)
        return [o.asnumpy() for o in self._outs]


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
        key = (val, type)
        if key not in self.constant_vars:
            name = f"_cst{val}{type}"
            self.constants[name] = np.array([val], dtype=nnvm_type,
                                            copy=False, ndmin=1)
            self.constant_vars[key] = sym.Variable(name)
            self.types[name] = nnvm_type
            self.shapes[name] = (1,)
        return self.constant_vars[key]

    def ref(self, n):
        """Resolve a reference to a node."""
        def setn(name, n):
            """Associate name with n."""
            self.eqv[n] = sym.Variable(name)
            if ismyiatype(n.type, Array):
                self.types[name] = nnvm_type_map(n.type.elements)
                self.shapes[name] = ashape(n)
            elif n.is_constant_graph():  # pragma: no cover
                raise Exception("This isn't tested")
                self.types[name] = 'int64'
                self.shapes[name] = (1,)
            else:
                self.types[name] = nnvm_type_map(n.type)
                self.shapes[name] = (1,)

        if n.is_constant() and not n.is_constant_graph():
            name = f"cst{next(self.c)}"
            self.constants[name] = np.array([n.value],
                                            dtype=type_to_np_dtype(n.type),
                                            copy=False, ndmin=1)
            setn(name, n)
        elif n not in self.eqv:
            name = f"i{next(self.c)}"
            self.inputs.append(n)
            self.input_names.append(name)
            setn(name, n)
        return self.eqv[n]

    def convert(self, lst, *, target='cpu', dev_id=0):
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

        if target == 'cpu':
            target = 'llvm'

        g = nnvm.graph.create(sym.Group(list(self.eqv[o] for o in outputs)))
        dg, lib, params = nnvm.compiler.build(
            g, target=target, shape=self.shapes, dtype=self.types,
            params=self.constants)

        shape = dg.json_attr('shape')
        types = dg.json_attr('dtype')
        index = dg.index

        def spec(entry_id):
            return (shape[entry_id],
                    graph_attr.TCODE_TO_DTYPE[types[entry_id]])

        output_specs = [spec(index.entry_id(x)) for x in index.output_entries]
        assert len(output_specs) == len(outputs)

        if target == 'llvm':
            context = tvm.cpu(dev_id)
        elif target == 'cuda':  # pragma: no cover
            context = tvm.gpu(dev_id)
        else:  # pragma: no cover
            raise Exception(f"Unsupported target: {target}")

        module = graph_runtime.create(dg, lib, context)

        for n, p in params.items():
            module.set_input(n, p)

        input_types = [self.types[i] for i in self.input_names]
        return (NNVMRunner(module, self.input_names,
                           input_types, output_specs, context),
                self.inputs, outputs)


converter = NNVMConverter(simple_map=SIMPLE_MAP, complex_map=COMPLEX_MAP)
nnvm_convert = converter.convert
