from ..ir import is_apply, is_constant, Graph, manage
from ..prim import Primitive, vm_implementations
from ..vm import VM

from myia.debug.buche import buche


def debug_convert(lst):
    """Returns graph, inputs, output, constants"""
    eqv = {}
    inputs = []

    g = Graph()

    def ref(n):
        if is_constant(n):
            pass
        elif n not in eqv:
            inputs.append(n)
            eqv[n] = g.add_parameter()
        return eqv[n]

    for n in lst:
        assert is_apply(n)
        assert is_constant(n.inputs[0], Primitive)
        fn = n.inputs[0].value
        args = [ref(a) for a in n.inputs[1:]]
        eqv[n] = g.apply(fn, *args)

    # TODO: figure out multiple outputs
    out = lst[-1]
    g.output = eqv[out]

    mng = manage(g)

    vm = VM(convert=lambda x: x, manager=mng, py_implementations={},
            implementations=vm_implementations)

    fn = vm.export(g)
    def wrap(*args):
        return [fn(*args)]

    return wrap, inputs, [out]
