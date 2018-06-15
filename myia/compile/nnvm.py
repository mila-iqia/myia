import numpy as np

import nnvm.compiler
import nnvm.symbol as sym
import tvm
from nnvm.compiler import graph_attr
from tvm.contrib import graph_runtime

from ..dtype import type_to_np_dtype
from ..ir import is_apply, is_constant
from ..prim import Primitive
from ..prim import ops as P

PRIMITIVE_MAP = {
    P.add: sym.elemwise_add,
    P.sub: sym.elemwise_sub,
    P.mul: sym.elemwise_mul,
    P.div: sym.elemwise_div,
    # P.mod:
    # P.pow:
    P.uadd: lambda x: x,
    P.usub: sym.negative,

    # P.eq:
    # P.lt:
    # P.gt:
    # P.ne:
    # P.le:
    # P.ge:
    # P.not_:

    # P.shape:
    # P.map_array:
    # P.scan_array:
    # P.reduce_array:
    # P.distribute:
    # P.reshape:
    # P.dot:
    }


def counter():
    val = -1

    def next():
        nonlocal val
        val += 1
        return val
    return next


def map_type(typ):
    return type_to_np_dtype(typ)


class NNVMRunner:
    def __init__(self, mod, input_names, output_specs):
        self.mod = mod
        self.input_names = input_names
        self.output_specs = output_specs
        self._outs = [tvm.nd.empty(spec[0], dtype=spec[1])
                      for spec in self.output_specs]

    def __call__(self, *args):
        assert len(args) == len(self.input_names)
        nnvm_args = dict(zip(self.input_names, args))
        self.mod.set_input(**nnvm_args)
        self.mod.run()
        for i, out in enumerate(self._outs):
            out = self.mod.get_output(i, out)
        return [o.asnumpy() for o in self._outs]


def nnvm_convert(lst):
    """Returns graph, inputs, output, constants"""
    eqv = {}
    inputs = []
    input_names = []
    constants = {}
    shapes = {}
    types = {}
    c = counter()

    def ref(n):
        if is_constant(n):
            name = f"cst{c()}"
            constants[name] = np.asarray(n.value, dtype=map_type(n.type))
            eqv[n] = sym.Variable(name)
            types[name] = map_type(n.type)
            shapes[name] = (1,)
        elif n not in eqv:
            name = f"i{c()}"
            inputs.append(n)
            input_names.append(name)
            eqv[n] = sym.Variable(name)
            types[name] = map_type(n.type)
            shapes[name] = (1,)
        return eqv[n]

    for n in lst:
        assert is_apply(n)
        assert is_constant(n.inputs[0], Primitive)
        fn = n.inputs[0].value
        args = [ref(a) for a in n.inputs[1:]]
        conv = PRIMITIVE_MAP.get(fn, None)
        if conv is not None:
            eqv[n] = conv(*args)
        else:
            raise NotImplementedError(fn)

    # TODO: figure out multiple outputs
    out = lst[-1]

    g = nnvm.graph.create(eqv[out])
    dg, lib, params = nnvm.compiler.build(
        g, target="llvm", shape=shapes, dtype=types, params=constants)

    shape = dg.json_attr('shape')
    types = dg.json_attr('dtype')
    index = dg.index

    def spec(entry_id):
        return (shape[entry_id], graph_attr.TCODE_TO_DTYPE[types[entry_id]])

    output_specs = [spec(index.entry_id(x)) for x in index.output_entries]

    module = graph_runtime.create(dg, lib, tvm.cpu())

    for n, p in params.items():
        module.set_input(n, p)

    print('nnvm_graph = ', dg.ir())
    return NNVMRunner(module, input_names, output_specs), inputs, [out]
