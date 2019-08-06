"""Graph generation from number of arguments or type signatures."""


from .clone import GraphCloner
from .manager import GraphManager
from .anf import Parameter, Graph

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


class ParametricGraph(Graph):
    """Graph with default arguments and varargs."""

    def __init__(self):
        """Initialize a ParametricGraph."""
        super().__init__()
        self.vararg = False

    def make_new(self, relation='copy'):
        """Make a new graph that's about this one."""
        g = super().make_new(relation)
        g.vararg = self.vararg
        return g

    def make_signature(self, args):
        """Create a signature which is the number of arguments."""
        return len(args)

    def generate_graph(self, nargs):
        """Generate a valid graph for the given number of arguments."""
        if self.vararg:
            actual_parameters = self.parameters[:-1]
            vararg = self.parameters[-1]
        else:
            actual_parameters = self.parameters
            vararg = None

        all_defaults = self.return_.inputs[2:]

        diff = nargs - (len(actual_parameters) - len(all_defaults))
        if diff < 0:
            raise MyiaTypeError(f'Not enough arguments for {self}')

        nvar = diff - len(all_defaults)
        if nvar > 0 and not vararg:
            raise MyiaTypeError(f'Too many arguments for {self}')

        cl = GraphCloner(self, total=True, graph_repl={self: Graph()})
        new_graph = cl[self]
        mng = GraphManager(manage=False, allow_changes=True)
        mng.add_graph(new_graph)
        new_parameters = [cl[p] for p in actual_parameters[:nargs]]

        defaulted_params = actual_parameters[nargs:]
        defaults = all_defaults[-len(defaulted_params):]
        for param, dflt in zip(defaulted_params, defaults):
            mng.replace(cl[param], cl[dflt])

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
        new_graph.return_.inputs[2:] = []
        return new_graph
