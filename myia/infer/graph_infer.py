"""Inference engine for Myia graphs."""

import asyncio
from types import FunctionType

from ..dtype import ismyiatype, Function
from ..debug.label import label
from ..ir import GraphGenerationError
from ..utils import Partializable, UNKNOWN, eprint, as_frozen

from .core import InferenceLoop, EvaluationCache, reify
from .utils import ANYTHING, InferenceError, MyiaTypeError, \
    infer_trace, Unspecializable, DEAD, POLY


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f"Wrong number of arguments for '{label(ident)}':"
        f" expected {expected}, got {got}."
    )


class TypeDispatchError(MyiaTypeError):
    """Represents an error in type dispatch for a MetaGraph."""

    def __init__(self, metagraph, types, refs=[], app=None):
        """Initialize a TypeDispatchError."""
        message = f'`{metagraph}` is not defined for argument types {types}'
        super().__init__(message, refs=refs, app=app)
        self.metagraph = metagraph
        self.types = types

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        eprint(f'{type(self).__qualname__}: {self.message}')


#########
# Track #
#########


class Track(Partializable):
    """Represents a property to infer."""

    def __init__(self, engine, name):
        """Initialize a Track."""
        self.engine = engine
        self.name = name

    def to_element(self, v):
        """Returns the value on this track for each element of v."""
        raise NotImplementedError()  # pragma: no cover

    def from_value(self, v, context=None):
        """Get the property from a value in the context."""
        raise NotImplementedError()  # pragma: no cover

    def default(self, values):
        """Default value for this track, if nothing is known."""
        raise NotImplementedError()  # pragma: no cover


##############
# References #
##############


class Context:
    """A context for the evaluation of a node.

    A context essentially contains the values of each relevant property of each
    parameter of each graph in which a node is nested.
    """

    @classmethod
    def empty(cls):
        """Create an empty context."""
        return Context(None, None, ())

    def __init__(self, parent, g, argkey):
        """Initialize the Context."""
        assert argkey is not None
        self.parent = parent
        self.graph = g
        self.argkey = argkey
        self.parent_cache = dict(parent.parent_cache) if parent else {}
        self.parent_cache[g] = self
        self._hash = hash((self.parent, self.graph, self.argkey))

    def filter(self, graph):
        """Return a context restricted to a graph's dependencies."""
        rval = self.parent_cache.get(graph, None)
        if rval is None:
            rval = self.parent_cache.get(graph.parent, None)
        return rval

    def add(self, graph, argkey):
        """Extend this context with values for another graph."""
        parent = self.parent_cache.get(graph.parent, None)
        return Context(parent, graph, argkey)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return type(other) is Context \
            and self.parent == other.parent \
            and self.graph == other.graph \
            and self.argkey == other.argkey


class Contextless:
    """Singleton Context which specializes to itself.

    CONTEXTLESS is essentially an empty context that is idempotent under
    all operations. In practice it maps each node of each graph to a unique
    type, shape and so on.
    """

    @classmethod
    def empty(cls):
        """Return CONTEXTLESS."""
        return CONTEXTLESS

    def filter(self, graph):
        """Return CONTEXTLESS."""
        return self

    def add(self, graph, argvals):
        """Return CONTEXTLESS."""
        return self


CONTEXTLESS = Contextless()


class AbstractReference:
    """Superclass for Reference and VirtualReference."""


class Reference(AbstractReference):
    """Reference to a certain node in a certain context.

    Attributes:
        engine: The InferenceEngine in which this Reference lives.
        node: The ANFNode being referred to.
        context: The Context for the node.

    """

    def __init__(self, engine, node, context):
        """Initialize the Reference."""
        self.node = node
        self.engine = engine
        g = node.value if node.is_constant_graph() else node.graph
        self.context = context and context.filter(g)
        self._hash = hash((self.node, self.context))

    async def __getitem__(self, track):
        """Get the value for the track (asynchronous)."""
        return await reify(await self.get_raw(track))

    def get(self, track="*"):
        """Get the value for the track (synchronous).

        If track is "*", return a dictionary with all tracks.
        """
        return self.engine.run_coroutine(self[track], throw=True)

    def get_raw(self, track):
        """Get the raw value for the track, which might be wrapped."""
        return self.engine.get_inferred(track, self)

    def __eq__(self, other):
        return isinstance(other, Reference) \
            and self.node is other.node \
            and self.context == other.context

    def __hash__(self):
        return self._hash


class VirtualReference(AbstractReference):
    """Synthetic reference that can be given to an inferrer.

    A VirtualReference contains the values it is supposed to take on
    every track, so `engine.get(track, vr)` returns `vr.values[track]`.

    Attributes:
        values: The values for that reference on each track.

    """

    def __init__(self, values):
        """Initialize the VirtualReference."""
        self.values = values

    async def get_raw(self, track):
        """Get the raw value for the track, which might be wrapped."""
        return self.values[track]

    async def __getitem__(self, track):
        return self.values[track]

    def __hash__(self):
        return hash(tuple(sorted(self.values.items())))

    def __eq__(self, other):
        return isinstance(other, VirtualReference) \
            and self.values == other.values


########
# Core #
########


class InferenceEngine:
    """Infer various properties about nodes in graphs.

    Arguments:
        tracks: Map each track (property name) to a Track object.
        tied_tracks: A dictionary from track names to lists of
            track names which should be computed along with it.
            E.g. tied_tracks={'type': ['shape']} to compute the
            shape every time the type is computed.

    """

    def __init__(self,
                 pipeline,
                 *,
                 tracks,
                 tied_tracks={},
                 context_class=Context):
        """Initialize the InferenceEngine."""
        self.loop = InferenceLoop()
        self.pipeline = pipeline
        self.mng = self.pipeline.resources.manager
        self.all_track_names = tuple(tracks.keys())
        self.tracks = {
            name: t(engine=self, name=name)
            for name, t in tracks.items()
        }
        self.tied_tracks = tied_tracks
        self.cache = EvaluationCache(loop=self.loop, keycalc=self.compute_ref)
        self.errors = []
        self.context_class = context_class
        self.reference_map = {}

    def run(self, graph, *, tracks, argspec, outspec=None):
        """Run the inferrer on a graph given initial values.

        Arguments:
            graph: The graph to analyze.
            tracks: The names of the tracks to infer.
            argspec: The arguments. Must be a tuple of dictionaries where
                each dictionary maps track name to value.
            outspec (optional): Expected inference results. If provided,
                inference results will be checked against them.
        """
        assert 'abstract' in self.tracks
        argrefs = [self.vref(arg) for arg in argspec]
        argspec = [ref.values['abstract'] for ref in argrefs]

        self.mng.add_graph(graph)
        empty_context = self.context_class.empty()
        root_context = empty_context.add(graph, as_frozen(argspec))
        output_ref = self.ref(graph.return_, root_context)

        async def _run():
            from ..abstract.inf import GraphXInferrer
            inf = GraphXInferrer(graph, empty_context)
            self.loop.schedule(
                inf(self.tracks['abstract'], None, argrefs)
            )

        async def _check():
            from ..abstract.inf import amerge
            amerge(await output_ref['abstract'], outspec['abstract'], loop=self.loop, forced=False)

        self.run_coroutine(_run())
        if outspec is not None:
            self.run_coroutine(_check())

        results = {name: output_ref.get(name) for name in tracks}
        return results, root_context

    def ref(self, node, context):
        """Return a Reference to the node in the given context."""
        return Reference(self, node, context)

    def vref(self, values):
        """Return a VirtualReference using the given property values."""
        return VirtualReference(values)

    async def compute_ref(self, key):
        """Compute the value of the Reference on the given track."""
        track_name, ref = key
        track = self.tracks[track_name]

        assert isinstance(ref, Reference)

        node = ref.node
        inferred = ref.node.inferred.get(track_name, UNKNOWN)

        if inferred is not UNKNOWN:
            inferred = track.from_external(inferred)
            return inferred

        elif node.is_constant():
            return await track.infer_constant(ref)

        elif node.is_apply():
            return await track.infer_apply(ref)

        else:
            raise AssertionError(f'Missing information for {key}', key)

    def get_inferred(self, track, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached.
        """
        return self.cache.get((track, ref))

    async def forward_reference(self, track, orig, new):
        self.reference_map[orig] = new
        return await self.get_inferred(track, new)

    def run_coroutine(self, coro, throw=True):
        """Run an async function using this inferrer's loop."""
        errs_before = len(self.errors)
        try:
            fut = self.loop.schedule(coro)
            self.loop.run_forever()
            self.errors.extend(self.loop.collect_errors())
            for err in self.errors[errs_before:]:
                err.engine = self
            if errs_before < len(self.errors):
                if throw:  # pragma: no cover
                    for err in self.errors:
                        if isinstance(err, InferenceError):
                            raise err
                    else:
                        raise err
                else:
                    return None  # pragma: no cover
            return fut.result()
        finally:
            for task in asyncio.all_tasks(self.loop):
                task._log_destroy_pending = False
