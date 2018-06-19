"""Linear implementation using the debug VM."""

from .utils import get_outputs

from ..ir import Graph, is_apply, is_constant, manage
from ..prim import Primitive, vm_implementations
from ..prim.ops import cons_tuple
from ..vm import VM


def make_tuple(vals, g):
    tup = g.constant(())
    for v in reversed(list(vals)):
        tup = g.apply(cons_tuple, v, tup)
    return tup


def debug_convert(lst):
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
        This implementation will convert the nodes into a subgraph
        that will run using the debug VM to help testing.

    """
    eqv = {}
    inputs = []
    outputs = []

    g = Graph()

    def ref(n):
        if is_constant(n):
            eqv[n] = n
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

    outputs = get_outputs(lst, lst[0].graph.manager.uses, set(eqv.keys()))
    g.output = make_tuple((eqv[o] for o in outputs), g)

    mng = manage(g)

    vm = VM(convert=lambda x: x, manager=mng, py_implementations={},
            implementations=vm_implementations)

    fn = vm.export(g)

    return fn, inputs, outputs
