"""Linear implementation using the debug VM."""

from ..ir import Graph, is_apply, is_constant, manage
from ..prim import Primitive, vm_implementations
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
