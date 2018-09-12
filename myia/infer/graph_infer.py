"""Inference engine for Myia graphs."""

import asyncio
from types import FunctionType

from ..dtype import ismyiatype, Function
from ..debug.label import label
from ..ir import GraphGenerationError
from ..utils import Partializable, UNKNOWN, eprint, as_frozen

from .core import InferenceLoop, EvaluationCache, EquivalenceChecker, reify, \
    reify_shallow
from .utils import ANYTHING, InferenceError, MyiaTypeError, DynamicMap, \
    infer_trace, Unspecializable, DEAD, POLY, AMBIGUOUS


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

    async def infer_constant(self, ctref):
        """Get the property for a ref of a Constant node."""
        return self.from_value(ctref.node.value, ctref.context)

    async def infer_apply(self, ref):
        """Get the property for a ref of an Apply node."""
        ctx = ref.context
        n_fn, *n_args = ref.node.inputs
        # We await on the function node to get the inferrer
        fn_ref = self.engine.ref(n_fn, ctx)
        inf = await fn_ref[self.name]
        if inf is ANYTHING:
            return ANYTHING

        argrefs = [self.engine.ref(node, ctx) for node in n_args]

        if not isinstance(inf, Inferrer):
            raise MyiaTypeError(
                f'Trying to call a non-callable type: {inf}',
                refs=[fn_ref],
                app=ref
            )

        return await self.engine.loop.schedule(
            inf(*argrefs),
            context_map={
                infer_trace: {**infer_trace.get(), ctx: ref}
            }
        )

    def to_element(self, v):
        """Returns the value on this track for each element of v."""
        raise NotImplementedError()  # pragma: no cover

    def from_value(self, v, context=None):
        """Get the property from a value in the context."""
        raise NotImplementedError()  # pragma: no cover

    def from_external(self, v):
        """Convert a property provided outside the inferrer."""
        return v

    def broaden(self, v):
        """Broaden the value for use in a graph's signature."""
        return v

    def default(self, values):
        """Default value for this track, if nothing is known."""
        raise NotImplementedError()  # pragma: no cover

    def assert_same(self, *vals, refs=[]):
        """Assert that all vals are the same on this track."""
        futs = [ref.get_raw(self.name)
                if isinstance(ref, AbstractReference) else ref
                for ref in vals]
        return self.engine.equiv.assert_same(*futs, refs=refs)

    def apply_predicate(self, predicate, res):
        """Apply a predicate on a value.

        The predicate can be a type, a callable, or a tuple of these
        things.
        """
        if isinstance(predicate, tuple):
            return any(self.apply_predicate(p, res) for p in predicate)
        elif ismyiatype(predicate):
            return ismyiatype(res, predicate)
        elif callable(predicate):
            return predicate(res)
        else:
            raise ValueError(predicate)  # pragma: no cover

    def predicate_error(self, predicate, res, ref):
        """Raise an error for the given predicate when a value doesn't match.

        Arguments:
            predicate: The predicate.
            res: The value that did not match the predicate.
            ref: The reference that produced the value.
        """
        def is_type(preds):
            return any(ismyiatype(pred) for pred in preds)

        if not isinstance(predicate, tuple):
            predicate = (predicate,)

        def _str(p):
            if isinstance(p, FunctionType):
                return p.__doc__
            else:
                return str(p)

        descrs = [_str(p) for p in predicate]
        if len(descrs) == 1:
            expected, = descrs
        else:  # pragma: no cover
            expected = ", ".join(descrs[:-1]) + ' or ' + descrs[-1]

        if is_type(predicate):
            err_cls = MyiaTypeError
        else:
            err_cls = InferenceError

        return err_cls(
            f'Expected: {expected}; Got: {res}',
            [ref]
        )

    async def check(self, predicate, *refs, return_tuple=False):
        """Assert that all refs match predicate, return values.

        This differs from will_check by resolving the values of each
        ref immediately, and returning them all instead of whichever
        resolves first.

        Arguments:
            predicate: A type that all refs must have, or a predicate
                function, or a tuple of types/predicates.
            refs: The references to compare.
            return_tuple: Whether to always return a tuple or not.
        """
        coros = [ref.get_shallow(self.name) for ref in refs]
        results = await asyncio.gather(*coros, loop=self.engine.loop)

        for ref, res in zip(refs, results):
            if not self.apply_predicate(predicate, res):
                raise self.predicate_error(
                    predicate,
                    res,
                    ref
                )

        if len(results) == 1 and not return_tuple:
            results, = results
        return results

    async def will_check(self, predicate, *refs):
        """Check that all refs match the predicate and each other.

        Checks are asynchronous and the value for one of the refs is
        returned (whichever is resolved first). That value may be
        an InferenceVar, which is guaranteed to *eventually* resolve to
        the same thing as all the other refs.
        """
        async def chk(ref):
            res = await ref[self.name]
            if not self.apply_predicate(predicate, res):
                raise self.predicate_error(
                    predicate,
                    res,
                    ref
                )
        for ref in refs:
            self.engine.loop.schedule(chk(ref))
        return await self.assert_same(*refs, refs=refs)


####################
# Inferrer classes #
####################


async def _concretize_type_helper(t, argrefs=None):
    if isinstance(t, Inferrer):
        return await t.as_function_type(argrefs)
    else:
        return await reify(t)


async def concretize_type(ref, argrefs=None):
    """Return the type of ref, resolving all Inferrer instances.

    Inferrer instances are resolved to Function types if possible. If an
    error occurs, this will return a Problem type.

    Arguments:
        ref: Either a Reference or an Inferrer.
        argrefs: References for the arguments passed to the Inferrer at
            the relevant call site.
    """
    t = (await ref['type']) if isinstance(ref, AbstractReference) else ref
    return await _concretize_type_helper(t, argrefs)


class Inferrer(DynamicMap):
    """Infer a property of the output of an operation.

    Attributes:
        engine: The InferenceEngine used by this inferrer.
        identifier: A Reference, Primitive, Graph, etc. that identifies
            the operation this Inferrer is about, for debugging purposes.
        cache: A cache of arguments (References) given to the operation,
            mapped to the result of the inference.

    """

    def __init__(self, track, identifier):
        """Initialize the Inferrer."""
        super().__init__()
        self.track = track
        self.engine = track.engine
        self.identifier = identifier

    async def get_unique_argrefs(self):
        """If possible, return a unique tuple of argrefs this was called on.

        Raises Unspecializable if it is not possible to get a single tuple
        of argrefs:

          * Unspecializable(DEAD) if the Inferrer was never called.
          * Unspecializable(POLY) if the Inferrer was called with at least
            two incompatible tuple of argrefs.
          * Unspecializable(AMBIGUOUS) in some obscure edge cases that we
            currently do not concern ourselves with.
        """
        # The cache works using References, but if two references have
        # the same inferred type/value/etc., we can merge their entries.
        cache = {}
        for x, y in self.cache.items():
            y = await reify(y)
            key = tuple([await arg[track] for arg in x
                         for track in self.engine.tracks])
            if key in cache and cache[key] != y:
                # NOTE: It's not completely clear when/why this tends to
                # happen. It seems to happen for PartialInferrers when
                # the differentiation is downstream, so in practice the
                # Problem node caused by this exception does not end up
                # in the final graph.
                raise Unspecializable(AMBIGUOUS)
            cache[key] = y

        if len(cache) == 0:
            raise Unspecializable(DEAD)
        elif len(cache) == 1:
            (argrefs, res), *_ = self.cache.items()
            return argrefs
        else:
            raise Unspecializable(POLY)

    async def as_function_type(self, argrefs=None):
        """Return a Function type corresponding to this Inferrer.

        Raises Unspecializable if this is not possible, e.g. if argrefs
        are not given and the Inferrer was called multiple times on
        different types.

        Arguments:
            argrefs: The argrefs the Inferrer was called on at the
                call site for which we need a Function. If None,
                we try to find suitable argrefs.
        """
        if argrefs is None:
            try:
                argrefs = await self.get_unique_argrefs()
            except Unspecializable as e:
                return e.args[0]
        cached = self.cache[tuple(argrefs)]
        return Function[[await concretize_type(argref)
                         for argref in argrefs],
                        await _concretize_type_helper(cached)]


class PrimitiveInferrer(Inferrer):
    """Infer a property of the result of a primitive.

    Attributes:
        inferrer_fn: A function to infer a property of the result of a
            primitive.

    """

    def __init__(self, track, prim, nargs, inferrer_fn):
        """Initialize the PrimitiveInferrer."""
        super().__init__(track, prim)
        self.inferrer_fn = inferrer_fn
        self.nargs = nargs

    def infer(self, *args):
        """Infer a property of the operation on the given arguments."""
        if self.nargs is not None and len(args) != self.nargs:
            raise type_error_nargs(self.identifier, self.nargs, len(args))
        return self.inferrer_fn(self.track, *args)

    def provably_equivalent(self, other):
        """Whether this inferrer is provably equivalent to the other.

        Two PrimitiveInferrers are equivalent if they use the same
        inferrer function.
        """
        return isinstance(other, PrimitiveInferrer) \
            and other.nargs == self.nargs \
            and other.inferrer_fn == self.inferrer_fn


class GraphInferrer(Inferrer):
    """Infer a property of the result of calling a Graph."""

    def __init__(self, track, graph, context, broaden=True):
        """Initialize the GraphInferrer."""
        super().__init__(track, graph)
        self.track = track
        self._graph = graph
        self.broaden = broaden
        if context is None:
            self.context = self.engine.context_class.empty()
        else:
            self.context = context.filter(graph)

    async def make_graph(self, args):
        """Return the graph to use for the given args."""
        return self._graph

    async def _make_argkey_and_context(self, args):
        g = await self.make_graph(args)
        argvals = []
        for arg in args:
            argval = {}
            for track_name, track in self.engine.tracks.items():
                result = await self.engine.get_inferred(track_name, arg)
                if self.broaden and not g.flags.get('flatten_inference'):
                    result = track.broaden(result)
                argval[track_name] = result
            argvals.append(argval)

        # Update current context using the fetched properties.
        argkey = as_frozen(argvals)
        return argkey, self.context.add(g, argkey)

    async def make_context(self, args):
        """Create a Context object for this graph with these arguments.

        We await on all relevant properties of all arguments in order to
        build a context (this cannot be done lazily, like with primitives,
        because we need a concrete context).
        """
        _, ctx = await self._make_argkey_and_context(args)
        return ctx

    async def infer(self, *args):
        """Infer a property of the operation on the given arguments."""
        g = await self.make_graph(args)
        nargs = len(g.parameters)

        if len(args) != nargs:
            raise type_error_nargs(self.identifier, nargs, len(args))

        argkey, context = await self._make_argkey_and_context(args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(g.parameters, argkey):
            for track, v in arg:
                ref = self.engine.ref(p, context)
                self.engine.cache.set_value((track, ref), v)

        out = self.engine.ref(g.return_, context)
        return await self.engine.get_inferred(self.track.name, out)

    def provably_equivalent(self, other):
        """Whether this inferrer is provably equivalent to the other.

        Two GraphInferrers are equivalent if they infer the same property
        on the same graph in the same context.
        """
        return isinstance(other, GraphInferrer) \
            and other.track == self.track \
            and other._graph == self._graph \
            and other.context == self.context


class MetaGraphInferrer(GraphInferrer):
    """Infer a property of the result of calling a MetaGraph."""

    def __init__(self, track, metagraph):
        """Initialize the MetaGraphInferrer."""
        super().__init__(track, metagraph, None)

    async def make_graph(self, args):
        """Return the graph to use for the given args."""
        types = [await arg['type'] for arg in args]
        try:
            return self._graph.specialize(
                self.engine.pipeline.resources, types
            )
        except GraphGenerationError as err:
            raise TypeDispatchError(self._graph, types) from None


class PartialInferrer(Inferrer):
    """Infer a property on a partial.

    This wraps another inferrer and defers all the work to it,
    prepending some arguments to all calls.
    """

    def __init__(self, track, fn, args):
        """Initialize the PartialInferrer."""
        super().__init__(track, 'partial')
        self.fn = fn
        self.args = tuple(args)

    async def infer(self, *args):
        """Add the partial arguments and defer to the wrapped inferrer."""
        return await self.fn(*(self.args + args))

    def provably_equivalent(self, other):
        """Wether this inferrer is equivalent to another.

        Two PartialInferrers are equivalent if the wrap the same
        inferrer and add the same arguments.
        """
        return (isinstance(other, PartialInferrer) and
                self.args == other.args and
                self.fn.provably_equivalent(other.fn))


class ExplicitInferrer(Inferrer):
    """Requires specific input types and returns a specific output type."""

    def __init__(self, track, argvals, retval):
        """Initialize ExplicitInferrer."""
        super().__init__(track, None)
        self.argvals = argvals
        self.retval = retval
        refs = []
        for v in argvals:
            # Work around having to find defaults for the other tracks
            # TODO: check if this causes problems
            d = {name: ANYTHING for name in self.engine.tracks.keys()}
            d[track.name] = v
            refs.append(VirtualReference(d))
        self.cache[tuple(refs)] = retval

    async def infer(self, *args):
        """Check arguments and return return type."""
        ngot = len(args)
        nexpect = len(self.argvals)
        if ngot != nexpect:
            raise MyiaTypeError(
                'Wrong number of arguments.'
                f' Expected {nexpect}, got {ngot}.',
                refs=[],
            )
        for got, aref in zip(self.argvals, args):
            self.engine.equiv.declare_equivalent(
                got,
                aref[self.track.name],
                refs=[aref]
            )
        return self.retval

    async def as_function_type(self, argrefs=None):
        """Return a Function type corresponding to this Inferrer."""
        return Function[self.argvals, self.retval]


def register_inferrer(*prims, nargs, constructors):
    """Define a PrimitiveInferrer for prims with nargs arguments.

    For each primitive, this registers a constructor for a PrimitiveInferrer
    that takes a track argument, in the constructors dictionary.
    """
    def deco(fn):
        def make_constructor(prim):
            def constructor(track):
                return PrimitiveInferrer(track, prim, nargs, fn)
            return constructor
        for prim in prims:
            constructors[prim] = make_constructor(prim)
        return fn
    return deco


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

    async def __reify__(self):
        """Return CONTEXTLESS."""
        return self

    def __str__(self):
        return 'CONTEXTLESS'

    def __repr__(self):
        return 'CONTEXTLESS'


CONTEXTLESS = Contextless()


class AbstractReference:
    """Superclass for Reference and VirtualReference."""

    def transform(self, fn):
        """Create a reference transformed through the given function."""
        return TransformedReference(self, fn)


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
        """Get the value for the track (asynchronous).

        If track is "*", return a dictionary with all tracks.
        """
        if track == '*':
            return {track: await self[track]
                    for track in self.engine.required_tracks}
        else:
            return await reify(await self.get_raw(track))

    def get(self, track="*"):
        """Get the value for the track (synchronous).

        If track is "*", return a dictionary with all tracks.
        """
        return self.engine.run_coroutine(self[track], throw=True)

    def get_raw(self, track):
        """Get the raw value for the track, which might be wrapped."""
        return self.engine.get_inferred(track, self)

    async def get_shallow(self, track):
        """Get the raw value for the track, which might be wrapped."""
        return await reify_shallow(await self.get_raw(track,))

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
        if track == '*':
            raise NotImplementedError('vref["*"]')
        else:
            return self.values[track]


class TransformedReference(AbstractReference):
    """Synthetic reference that modifies another reference.

    Attributes:
        ref: The original reference.
        fn: The function to call on (track, value) to modify the value
            inferred for ref.

    """

    def __init__(self, ref, fn):
        """Initialize the TransformedReference."""
        self.ref = ref
        self.fn = fn

    async def _parent_raw(self, track_name):
        track = self.ref.engine.tracks[track_name]
        return track, await self.ref.get_raw(track_name)

    async def get_raw(self, track_name):
        """Get the raw value for the track."""
        track, v = await self._parent_raw(track_name)
        return self.fn(track, v)

    async def __getitem__(self, track_name):
        if track_name == '*':
            raise NotImplementedError('tref["*"]')
        else:
            track, v = await self._parent_raw(track_name)
            return self.fn(track, await reify(v))


########
# Core #
########


class InferenceEngine:
    """Infer various properties about nodes in graphs.

    Arguments:
        tracks: Map each track (property name) to a Track object.
        required_tracks: A list of tracks that will be inferred for
            the output. Other tracks may be used if requested in
            the evaluation of a required track.
        eq_class: The class to use to check equivalence between
            values.

    """

    def __init__(self,
                 pipeline,
                 *,
                 tracks,
                 required_tracks=None,
                 eq_class=EquivalenceChecker,
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
        self.required_tracks = required_tracks or self.all_track_names
        self.cache = EvaluationCache(loop=self.loop, keycalc=self.compute_ref)
        self.errors = []
        self.equiv = eq_class(
            loop=self.loop,
            error_callback=self.errors.append
        )
        self.context_class = context_class

    def run(self, graph, argvals):
        """Run the inferrer on a graph given initial values.

        Arguments:
            graph: The graph to analyze.
            argvals: The arguments. Must be a tuple of dictionaries where
                each dictionary maps track name to value.
        """
        argrefs = [self.vref(arg) for arg in argvals]
        argvals = [{t: ref.values[t] for t in self.all_track_names}
                   for ref in argrefs]

        self.mng.add_graph(graph)
        empty_context = self.context_class.empty()
        root_context = empty_context.add(graph, as_frozen(argvals))
        output_ref = self.ref(graph.return_, root_context)

        async def _run():
            for track in self.required_tracks:
                inf = GraphInferrer(self.tracks[track],
                                    graph, empty_context,
                                    broaden=False)
                self.loop.schedule(inf(*argrefs))

        self.run_coroutine(_run())
        return output_ref.get(), root_context

    def ref(self, node, context):
        """Return a Reference to the node in the given context."""
        return Reference(self, node, context)

    def vref(self, values):
        """Return a VirtualReference using the given property values."""
        for track_obj in self.tracks.values():
            if track_obj.name not in values:
                values[track_obj.name] = track_obj.default(values)
        return VirtualReference(values)

    async def compute_ref(self, key):
        """Compute the value of the Reference on the given track."""
        track_name, ref = key
        track = self.tracks[track_name]

        if isinstance(ref, (VirtualReference, TransformedReference)):
            # A VirtualReference already contains the values we need.
            return await ref.get_raw(track_name)

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
            return track.default({})

    def invalidate(self, track, ref):
        """Invalidate the current key in the cache and return the old value.

        Raises KeyError if the key wasn't in the cache to begin with.
        """
        return self.cache.invalidate((track, ref))

    def get_inferred(self, track, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached.
        """
        return self.cache.get((track, ref))

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
                    raise self.errors[-1]
                else:
                    return None  # pragma: no cover
            return fut.result()
        finally:
            for task in asyncio.Task.all_tasks(self.loop):
                task._log_destroy_pending = False
