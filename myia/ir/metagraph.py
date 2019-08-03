"""Graph generation from number of arguments or type signatures."""


from .clone import GraphCloner
from .manager import GraphManager
from .anf import Parameter

from .. import abstract, parser
from ..info import About
from ..prim import ops as P
from ..utils import MyiaTypeError


class TypeDispatchError(MyiaTypeError):
    """Represents an error in type dispatch for a MetaGraph."""

    def __init__(self, metagraph, types, refs=[]):
        """Initialize a TypeDispatchError."""
        message = f'`{metagraph}` is not defined for argument types {types}'
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
        raise NotImplementedError('Override generate_graph in subclass.')

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


class ParametricGraph(MetaGraph):
    """Graph with default arguments and varargs."""

    def __init__(self, graph, defaults, vararg):
        """Initialize a ParametricGraph."""
        super().__init__(graph.debug.name)
        self.graph = graph
        self.defaults = defaults
        self.vararg = vararg

    def set_flags(self, *flags, **kwflags):
        """Set flags on the underlying graph."""
        self.graph.set_flags(*flags, **kwflags)

    def make_signature(self, args):
        """Create a signature which is the number of arguments."""
        return len(args)

    def generate_graph(self, nargs):
        """Generate a valid graph for the given number of arguments."""
        if self.vararg:
            actual_parameters = self.graph.parameters[:-1]
            vararg = self.graph.parameters[-1]
        else:
            actual_parameters = self.graph.parameters
            vararg = None

        diff = nargs - len(actual_parameters)
        if diff < 0:
            raise MyiaTypeError(f'Not enough arguments for {self}')

        cl = GraphCloner(self.graph, total=True)
        new_graph = cl[self.graph]
        mng = GraphManager(manage=False, allow_changes=True)
        mng.add_graph(new_graph)
        new_parameters = [cl[p] for p in actual_parameters]

        for param, node in self.defaults[:diff]:
            with About(param.debug, 'copy'):
                new_param = Parameter(new_graph)
            new_parameters.append(new_param)
            mng.replace(cl[node], new_param)

        nvar = diff - len(self.defaults)
        if nvar > 0 and not vararg:
            raise MyiaTypeError(f'Too many arguments for {self}')

        if vararg:
            v_parameters = []
            for i in range(nvar):
                new_param = Parameter(new_graph)
                new_param.debug.name = f'{vararg.debug.name}[{i}]'
                v_parameters.append(new_param)
            new_parameters += v_parameters
            constructed = new_graph.apply(
                P.make_tuple, *v_parameters
            )
            v2 = cl[vararg]
            mng.replace(v2, constructed)

        mng.set_parameters(new_graph, new_parameters)
        if mng.free_variables_total[new_graph]:
            raise Exception(
                'Graphs with default arguments or varargs cannot be closures'
            )
        return new_graph
