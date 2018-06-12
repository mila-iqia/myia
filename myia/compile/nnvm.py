import ..prim.ops as P
from ..ir import Apply, toposort, is_parameter, is_apply, is_constant_graph, is_constant, Parameter, manage
from ..prim import Primitive

import nnvm.compiler
import nnvm.symbol as sym


def counter():
    val = -1
    def next():
        nonlocal val
        val += 1
        return val
    return next


def nnvm_convert(lst):
    """Returns graph, inputs, output, constants"""
    eqv = {}
    inputs = []
    constants = []
    c = counter()

    def ref(n):
        if is_constant(n):
            name = f"cst{c()}"
            constants.append((name, n.value))
            eqv[n] = sym.Variable(name)
        elif n not in eqv:
            inputs.append(n)
            eqv[n] = sym.Variable(f"i{c()}")

        return eqv[n]

    for n in lst:
       assert is_apply(n)
       assert is_constant(n.inputs[0], Primitive)
       fn = n.inputs[0].value
       args = [ref(a) for a in n.inputs[1:]]
       if fn == add:
           eqv[n] = sym.elemwise_add(*args)
       else:
           raise NotImplementedError(fn)

    # TODO: figure out multiple outputs
    out = lst[-1]
    g = nnvm.graph.create(eqv[out])
    return g, inputs, constants, [out]
