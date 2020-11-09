"""Implementation of the 'universal' macro."""

from ovld import ovld

from .. import lib, operations
from ..lib import Constant, MetaGraph, core, macro
from ..utils import untested


@core(universal=True)
def universal_wrapper(fn):
    """Create a version of fn that passes the universe through."""

    def wrapped(*args):
        U = args[-1]
        args = args[:-1]
        return U, fn(*args)

    return wrapped


@ovld
def is_universal(self, g: lib.GraphFunction):
    """Check whether a function is universal or not."""
    return g.graph.has_flags("universal")


@ovld  # noqa: F811
def is_universal(self, g: lib.PartialApplication):
    return self(g.fn)


@ovld  # noqa: F811
def is_universal(self, p: (lib.PrimitiveFunction, lib.TypedPrimitive)):
    return p.prim.universal


@ovld  # noqa: F811
def is_universal(self, x: object):
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
        with untested():
            return info.graph.apply(universal_wrapper, fn.node)


__operation_defaults__ = {
    "name": "universal",
    "registered_name": "universal",
    "mapping": universal,
    "python_implementation": None,
}
