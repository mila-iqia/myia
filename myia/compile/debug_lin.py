"""Linear implementation using the debug VM."""

from .utils import get_outputs

from ..ir import Graph, manage, clone
from ..prim import Primitive, vm_implementations, ops as P
from ..vm import VM


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
        if n.is_constant():
            eqv[n] = n
        elif n not in eqv:
            inputs.append(n)
            eqv[n] = g.add_parameter()
        return eqv[n]

    for n in lst:
        assert n.is_apply()
        assert n.inputs[0].is_constant(Primitive)
        fn = n.inputs[0].value
        args = [ref(a) for a in n.inputs[1:]]
        eqv[n] = g.apply(fn, *args)

    outputs = get_outputs(lst, lst[0].graph.manager.uses, set(eqv.keys()))
    g.output = g.apply(P.make_tuple, *[eqv[o] for o in outputs])

    # Clone in case g contains subgraphs that have a different manager
    g = clone(g)

    mng = manage(g)

    vm = VM(convert=lambda x: x, manager=mng, py_implementations={},
            implementations=vm_implementations)

    fn = vm.export(g)

    return fn, inputs, outputs
