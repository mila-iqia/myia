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

    def __init__(self, engine):
        """Initialize a Track."""
        self.engine = engine


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

    async def get(self):
        """Get the value for the track (asynchronous)."""
        raw = self.engine.get_inferred(self)
        return await reify(await raw)

    def get_sync(self):
        """Get the value for the track (synchronous)."""
        return self.engine.run_coroutine(self.get(), throw=True)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Reference) \
            and self.node is other.node \
            and self.context == other.context


class VirtualReference(AbstractReference):
    """Synthetic reference that can be given to an inferrer.

    A VirtualReference contains the values it is supposed to take on
    every track, so `engine.get(track, vr)` returns `vr.values[track]`.

    Attributes:
        values: The values for that reference on each track.

    """

    def __init__(self, abstract):
        """Initialize the VirtualReference."""
        self.abstract = abstract

    async def get(self):
        """Get the value for the track (asynchronous)."""
        return self.abstract

    def get_sync(self):
        """Get the value for the track (synchronous)."""
        return self.abstract

    def __hash__(self):
        return hash(self.abstract)

    def __eq__(self, other):
        return isinstance(other, VirtualReference) \
            and self.abstract == other.abstract


########
# Core #
########


class InferenceEngine:
    """Infer various properties about nodes in graphs.

    Arguments:
        tracks: Map each track (property name) to a Track object.

    """

    def __init__(self,
                 pipeline,
                 *,
                 track,
                 context_class=Context):
        """Initialize the InferenceEngine."""
        self.loop = InferenceLoop()
        self.pipeline = pipeline
        self.mng = self.pipeline.resources.manager
        self.track = track(engine=self)
        self.cache = EvaluationCache(loop=self.loop, keycalc=self.compute_ref)
        self.errors = []
        self.context_class = context_class
        self.reference_map = {}

    def run(self, graph, *, argspec, outspec=None):
        """Run the inferrer on a graph given initial values.

        Arguments:
            graph: The graph to analyze.
            argspec: The arguments. Must be a tuple of AbstractBase.
            outspec (optional): Expected inference result. If provided,
                inference result will be checked against it.
        """
        from ..abstract.inf import GraphXInferrer
        from ..abstract.base import AbstractBase
        assert not isinstance(outspec, dict)
        argrefs = [VirtualReference(arg) for arg in argspec]

        self.mng.add_graph(graph)
        empty_context = self.context_class.empty()
        root_context = empty_context.add(graph, as_frozen(argspec))
        output_ref = self.ref(graph.return_, root_context)

        async def _run():
            inf = GraphXInferrer(graph, empty_context)
            self.loop.schedule(
                inf(self.track, None, argrefs)
            )

        async def _check():
            from ..abstract.inf import amerge
            amerge(await output_ref.get(),
                   outspec,
                   loop=self.loop,
                   forced=False)

        self.run_coroutine(_run())
        if outspec is not None:
            self.run_coroutine(_check())

        return output_ref.get_sync(), root_context

    def ref(self, node, context):
        """Return a Reference to the node in the given context."""
        return Reference(self, node, context)

    async def compute_ref(self, ref):
        """Compute the value of the Reference on the given track."""
        assert isinstance(ref, Reference)
        track = self.track

        node = ref.node
        inferred = ref.node.inferred.get('abstract', UNKNOWN)

        if inferred is not UNKNOWN:
            assert inferred is ref.node.abstract
            return inferred

        elif node.is_constant():
            return await track.infer_constant(ref)

        elif node.is_apply():
            return await track.infer_apply(ref)

        else:
            raise AssertionError(f'Missing information for {ref}', ref)

    def get_inferred(self, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached.
        """
        return self.cache.get(ref)

    async def forward_reference(self, orig, new):
        self.reference_map[orig] = new
        return await self.get_inferred(new)

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
