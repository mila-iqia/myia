"""Implementation of the 'universal' macro."""

from .. import lib, operations
from ..lib import Constant, MetaGraph, macro, overload


@overload
def is_universal(g: lib.GraphFunction):
    """Check whether a function is universal or not."""
    return g.graph.has_flags("universal")


@overload  # noqa: F811
def is_universal(x: object):
    return False


class StatePassthrough(MetaGraph):
    """Wrap a function to pass through the universe argument.

    The wrapped function returns the universe parameter unchanged.
    """

    def __init__(self, op):
        """Initialize a StatePassthrough."""
        super().__init__(f"U.{op}")
        self.op = op

    def expand(self, g, parameters):
        """Generate a graph wrapper based on the number of arguments."""
        *params, u = parameters
        res = g.apply(self.op, *params)
        return g.apply(operations.make_tuple, u, res)

    async def reroute(self, engine, outref, argrefs):
        """Inline the Graph/MetaGraph if it has the appropriate flag."""
        return engine.ref(
            self.expand(outref.node.graph, [arg.node for arg in argrefs]),
            outref.context,
        )


@macro
async def universal(info, fn):
    """Macro implementation for 'universal'."""
    absfn = await fn.get()
    options = await absfn.get()
    if all(is_universal(opt) for opt in options):
        return fn

    elif fn.node.is_constant():
        return Constant(StatePassthrough(fn.node.value))

    else:
        raise NotImplementedError(
            'Macro "universal" does not currently work on non-constant'
            " functions."
        )


__operation_defaults__ = {
    "name": "universal",
    "registered_name": "universal",
    "mapping": universal,
    "python_implementation": None,
}
