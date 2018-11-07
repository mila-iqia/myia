"""Linear implementation using NNVM."""

import numpy as np

import nnvm.compiler
import nnvm.symbol as sym
import tvm
from nnvm.compiler import graph_attr
from tvm.contrib import graph_runtime

from .utils import get_outputs

from ..dtype import type_to_np_dtype, ismyiatype, Array
from ..prim import Primitive
from ..prim import ops as P
from ..ir import Graph


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
    vshp = v.shape
    if len(shp) != len(vshp):
        # We need to pad the shape
        vshp = (1,) * (len(shp) - len(vshp)) + vshp
        nv = sym.reshape(nv, shape=vshp)
    if shp == vshp:
        return nv
    return sym.broadcast_to(nv, shape=shp)


def nnvm_dot(c, a, b):
    """Implementation of dot."""
    na = c.ref(a)
    nb = c.ref(b)
    return sym.dense(na, sym.transpose(nb), units=b.shape[1], use_bias=False)


def nnvm_array_map(c, fn, *array):
    """Implementation of array_map."""
    assert fn.is_constant()
    fn = fn.value
    if fn in SIMPLE_MAP:
        # This might go away at some point since we have wrap_primitives
        return SIMPLE_MAP[fn](*[c.ref(a) for a in array])  # pragma: no cover
    else:
        assert isinstance(fn, Graph)
        node = fn.output
        # Handle wrapping graphs
        if (len(fn.parameters) == (len(node.inputs) - 1) and
                node.inputs[0].is_constant() and
                tuple(node.inputs[1:]) == tuple(fn.parameters)):
            fn = node.inputs[0].value
            if fn in SIMPLE_MAP:
                return SIMPLE_MAP[fn](*[c.ref(a) for a in array])
        raise NotImplementedError("Only support primitives for array_map")


COMPLEX_MAP = {
    P.bool_not: nnvm_bool_not,
    P.distribute: nnvm_distribute,
    P.dot: nnvm_dot,
    P.array_map: nnvm_array_map,
}


def nnvm_type_map(type):
    """Map a numpy type to an NNVM type."""
    dt = type_to_np_dtype(type)
    if dt == 'bool':
        dt = 'uint8'
    return dt


def counter():
    """Returns a function that returns increasing numbers with each call."""
    val = -1

    def next():
        nonlocal val
        val += 1
        return val
    return next


class NNVMRunner:
    """Adapter to run an NNVM module."""

    def __init__(self, mod, input_names, input_types, output_specs):
        """Intialize the runner.

        Arguments:
            mod: NNVM compiled module
            input_names: list of the names of inputs (in order)
            output_specs: list of shape and dtype for outputs
                          [(shp0, dtype0), ...]

        """
        self.mod = mod
        self.input_names = input_names
        self.input_types = input_types
        self.output_specs = output_specs
        self._outs = [tvm.nd.empty(spec[0], dtype=spec[1])
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
                self.shapes[name] = n.shape
            elif n.is_constant_graph():  # pragma: no cover
                raise Exception("This isn't tested")
                self.types[name] = 'int64'
                self.shapes[name] = (1,)
            else:
                self.types[name] = nnvm_type_map(n.type)
                self.shapes[name] = (1,)

        if n.is_constant() and not n.is_constant_graph():
            name = f"cst{self.c()}"
            self.constants[name] = np.array([n.value],
                                            dtype=type_to_np_dtype(n.type),
                                            copy=False, ndmin=1)
            setn(name, n)
        elif n not in self.eqv:
            name = f"i{self.c()}"
            self.inputs.append(n)
            self.input_names.append(name)
            setn(name, n)
        return self.eqv[n]

    def convert(self, lst):
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
        self.c = counter()
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
        g = nnvm.graph.create(sym.Group(list(self.eqv[o] for o in outputs)))
        dg, lib, params = nnvm.compiler.build(
            g, target="llvm", shape=self.shapes, dtype=self.types,
            params=self.constants)

        shape = dg.json_attr('shape')
        types = dg.json_attr('dtype')
        index = dg.index

        def spec(entry_id):
            return (shape[entry_id],
                    graph_attr.TCODE_TO_DTYPE[types[entry_id]])

        output_specs = [spec(index.entry_id(x)) for x in index.output_entries]
        assert len(output_specs) == len(outputs)

        module = graph_runtime.create(dg, lib, tvm.cpu())

        for n, p in params.items():
            module.set_input(n, p)

        input_types = [self.types[i] for i in self.input_names]
        return (NNVMRunner(module, self.input_names,
                           input_types, output_specs),
                self.inputs, outputs)


converter = NNVMConverter(simple_map=SIMPLE_MAP, complex_map=COMPLEX_MAP)
nnvm_convert = converter.convert
