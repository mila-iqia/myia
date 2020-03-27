"""Graph generation from number of arguments or type signatures."""


from .. import abstract, parser
from ..utils import MyiaTypeError


class TypeDispatchError(MyiaTypeError):
    """Represents an error in type dispatch for a MetaGraph."""

    def __init__(self, metagraph, types, refs=[]):
        """Initialize a TypeDispatchError."""
        message = f"`{metagraph}` is not defined for argument types {types}"
        super().__init__(message, refs=refs)
        self.metagraph = metagraph
        self.types = types


class MetaGraph:
    """Graph generator.

    Can be called with a pipeline's resources and a list of argument types to
    generate a graph corresponding to these types.
    """

    def __init__(self, name):
        """Initialize a MetaGraph."""
        self.name = name

    async def normalize_args(self, args):
        """Return normalized versions of the arguments.

        By default, this returns args unchanged.
        """
        return self.normalize_args_sync(args)

    def normalize_args_sync(self, args):
        """Return normalized versions of the arguments.

        By default, this returns args unchanged.
        """
        return args

    def make_signature(self, args):
        """Return a signature corresponding to the args.

        Each signature corresponds to a graph.
        """
        return args

    def generate_graph(self, args):
        """Generate a Graph for the given abstract arguments."""
        raise NotImplementedError("Override generate_graph in subclass.")

    async def reroute(self, engine, outref, argrefs):
        """By default, MetaGraphs do not reroute."""
        return None

    def __str__(self):
        return self.name


class MultitypeGraph(MetaGraph):
    """Associates type signatures to specific graphs."""

    def __init__(self, name, entries={}):
        """Initialize a MultitypeGraph."""
        super().__init__(name)
        self.entries = list(entries.items())

    def normalize_args_sync(self, args):
        """Return broadened arguments."""
        return tuple(abstract.broaden(a) for a in args)

    def register(self, *types):
        """Register a function for the given type signature."""

        def deco(fn):
            atypes = tuple(abstract.type_to_abstract(t) for t in types)
            self.entries.append((atypes, fn))
            return fn

        return deco

    def _getfn(self, types):
        for sig, fn in self.entries:
            if abstract.typecheck(sig, types):
                return fn
        else:
            raise TypeDispatchError(self, types)

    def generate_graph(self, args):
        """Generate a Graph for the given abstract arguments."""
        return parser.parse(self._getfn(tuple(args)))

    def __call__(self, *args):
        """Call like a normal function."""
        types = tuple(abstract.to_abstract(arg) for arg in args)
        fn = self._getfn(types)
        return fn(*args)


__consolidate__ = True
__all__ = ["MetaGraph", "MultitypeGraph", "TypeDispatchError"]
