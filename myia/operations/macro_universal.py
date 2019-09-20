"""Implementation of the 'universal' macro."""

from .. import lib, operations, xtype
from ..lib import (
    Constant,
    Graph,
    MetaGraph,
    MyiaTypeError,
    Reference,
    macro,
    overload,
)
from ..parser import parse


@overload
def is_universal(g: lib.GraphFunction):
    return g.graph.has_flags('universal')


@overload
def is_universal(x: object):
    return False


@overload
def to_universal(gf: lib.GraphFunction):
    g = gf.graph
    if g.parent:
        raise NotImplementedError(f'Cannot universalize closure {g}')
    elif g.has_flags('original'):
        assert not g.has_flags('universal')
        return parse(g.flags['original'], use_universe=True)
    else:
        raise NotImplementedError(f'Cannot universalize {g}')


@overload
def to_universal(pf: lib.PrimitiveFunction):
    return StatePassthrough(pf.prim)


@overload
def to_universal(mf: lib.MacroFunction):
    return StatePassthrough(mf.macro)


@overload
def to_universal(mf: lib.MetaGraphFunction):
    return StatePassthrough(mf.metagraph)


@overload
def to_universal(x: object):
    raise NorImplementedError(f'Cannot universalize {x}')


class StatePassthrough(MetaGraph):
    """Wrap a function to pass through the universe argument.

    The wrapped function returns the universe parameter unchanged.
    """

    def __init__(self, op):
        """Initialize a StatePassthrough."""
        super().__init__(f'U.{op}')
        self.op = op

    async def normalize_args(self, args):
        """The args are passed through as they are."""
        return args

    def make_signature(self, args):
        """The signature is the number of arguments passed.

        We subtract one, because the first argument is the universe.
        """
        assert len(args) > 0
        return len(args) - 1

    def expand(self, g, parameters):
        """Generate a graph wrapper based on the number of arguments."""
        u, *params = parameters
        res = g.apply(self.op, *params)
        return g.apply(operations.make_tuple, u, res)

    def generate_graph(self, nargs):
        """Generate a graph wrapper based on the number of arguments."""
        g = Graph()
        g.debug.name = self.name
        u = g.add_parameter()
        u.debug.name = 'U'
        for _ in range(nargs):
            g.add_parameter()
        g.output = self.expand(g, g.parameters)
        return g

    async def reroute(self, engine, outref, argrefs):
        """Inline the Graph/MetaGraph if it has the appropriate flag."""
        return engine.ref(
            self.expand(outref.node.graph, [arg.node for arg in argrefs]),
            outref.context
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
        repl = {to_universal(opt) for opt in options}
        if len(repl) == 1:
            rval, = repl
            return Constant(rval)
        else:
            raise Exception(f'Too many options')


__operation_defaults__ = {
    'name': 'universal',
    'registered_name': 'universal',
    'mapping': universal,
    'python_implementation': None,
}
