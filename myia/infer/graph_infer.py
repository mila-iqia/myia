"""Inference engine for Myia graphs."""

import asyncio
from types import FunctionType

from ..dtype import Type, Function
from ..ir import is_constant, is_constant_graph, is_apply
from ..utils import Partializable, UNKNOWN

from .core import InferenceLoop, EvaluationCache, EquivalenceChecker, reify
from .utils import ANYTHING, InferenceError, MyiaTypeError, DynamicMap


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f'Wrong number of arguments for {ident}:'
        f' expected {expected}, got {got}.'
    )


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
        if isinstance(inf, Function):
            ngot = len(argrefs)
            nexpect = len(inf.arguments)
            if ngot != nexpect:
                raise MyiaTypeError(
                    'Wrong number of arguments.'
                    f' Expected {nexpect}, got {ngot}.',
                    refs=[],
                    app=ref
                )
            for got, aref in zip(inf.arguments, argrefs):
                expect = await aref[self.name]
                if expect != got:
                    raise MyiaTypeError(
                        'Type mismatch.'
                        f' Expected {expect}, got {got}.',
                        refs=[aref],
                        app=ref
                    )
            return inf.retval

        if not isinstance(inf, Inferrer):
            raise MyiaTypeError(
                f'Trying to call a non-callable type.',
                refs=[fn_ref],
                app=ref
            )

        try:
            return await inf(*argrefs)
        except InferenceError as infe:
            # This builds a traceback of sorts in traceback_refs
            # The first encounter with a ctx will be the caller,
            # the others will be subsequence operations that depend
            # on the result.
            infe.traceback_refs.setdefault(ctx, ref)
            raise

    def from_value(self, v, context=None):
        """Get the property from a value in the context."""
        raise NotImplementedError()  # pragma: no cover

    def broaden(self, v):
        """Broaden the value for use in a graph's signature."""
        return v

    def default(self, values):
        """Default value for this track, if nothing is known."""
        raise NotImplementedError()  # pragma: no cover

    def assert_same(self, *vals, refs=[]):
        """Assert that all vals are the same on this track."""
        futs = [ref.get_raw(self.name) if isinstance(ref, Reference) else ref
                for ref in vals]
        return self.engine.equiv.assert_same(*futs, refs=refs)

    def apply_predicate(self, predicate, res):
        """Apply a predicate on a value.

        The predicate can be a type, a callable, or a tuple of these
        things.
        """
        if isinstance(predicate, tuple):
            return any(self.apply_predicate(p, res) for p in predicate)
        elif isinstance(predicate, Type):
            return res == predicate
        elif isinstance(predicate, type) and issubclass(predicate, Type):
            return isinstance(res, predicate)
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
            return any(isinstance(pred, Type)
                       or isinstance(pred, type)
                       and issubclass(pred, Type)
                       for pred in preds)

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
        else:
            expected = ", ".join(descrs[:-1]) + ' or ' + descrs[-1]

        if is_type(predicate):
            err_cls = MyiaTypeError
        else:
            err_cls = InferenceError

        return err_cls(
            f'Expected: {expected}; Got: {res}',
            [ref]
        )

    async def check(self, predicate, *refs):
        """Assert that all refs match predicate, return values.

        This differs from will_check by resolving the values of each
        ref immediately, and returning them all instead of whichever
        resolves first.

        Arguments:
            predicate: A type that all refs must have, or a predicate
                function, or a tuple of types/predicates.
            refs: The references to compare.
        """
        coros = [ref[self.name] for ref in refs]
        results = await asyncio.gather(*coros, loop=self.engine.loop)

        for ref, res in zip(refs, results):
            if not self.apply_predicate(predicate, res):
                raise self.predicate_error(
                    predicate,
                    res,
                    ref
                )

        if len(results) == 1:
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
        return await self.assert_same(*refs)


####################
# Inferrer classes #
####################


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
    """Infer a property of the result of calling a Graph.

    Attributes:
        track: Name of the property to infer.
        graph: The Graph to infer on.
        context: The context for the given graph.

    """

    def __init__(self, track, graph, context):
        """Initialize the GraphInferrer."""
        super().__init__(track, graph)
        self.track = track
        self.graph = graph
        self.nargs = len(self.graph.parameters)
        self.context = context.filter(graph)

    async def make_context(self, args):
        """Create a Context object for this graph with these arguments.

        We await on all relevant properties of all arguments in order to
        build a context (this cannot be done lazily, like with primitives,
        because we need a concrete context).
        """
        argvals = []
        for arg in args:
            argval = {}
            for track_name, track in self.engine.tracks.items():
                result = await self.engine.get_inferred(track_name, arg)
                if not self.graph.flags.get('flatten_inference'):
                    result = track.broaden(result)
                argval[track_name] = result
            argvals.append(argval)

        # Update current context using the fetched properties.
        return self.context.add(self.graph, argvals)

    async def infer(self, *args):
        """Infer a property of the operation on the given arguments."""
        if len(args) != self.nargs:
            raise type_error_nargs(self.identifier, self.nargs, len(args))

        context = await self.make_context(args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(self.graph.parameters, context.argkey):
            for track, v in arg:
                ref = self.engine.ref(p, context)
                self.engine.cache.set_value((track, ref), v)

        out = self.engine.ref(self.graph.return_, context)
        return await self.engine.get_inferred(self.track.name, out)

    def provably_equivalent(self, other):
        """Whether this inferrer is provably equivalent to the other.

        Two GraphInferrers are equivalent if they infer the same property
        on the same graph in the same context.
        """
        return isinstance(other, GraphInferrer) \
            and other.track == self.track \
            and other.graph == self.graph \
            and other.context == self.context


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

    def infer(self, *args):
        """Add the partial arguments and defer to the wrapped inferrer."""
        return self.fn(*(self.args + args))

    def provably_equivalent(self, other):
        """Wether this inferrer is equivalent to another.

        Two PartialInferrers are equivalent if the wrap the same
        inferrer and add the same arguments.
        """
        return (isinstance(other, PartialInferrer) and
                self.args == other.args and
                self.fn.provably_equivalent(other.fn))


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

    def __init__(self, parent, g, argvals, raw_argvals=False):
        """Initialize the Context."""
        self.parent = parent
        self.graph = g
        if raw_argvals:
            self.argkey = argvals
        else:
            self.argkey = tuple(tuple(sorted(argv.items()))
                                for argv in argvals)
        self.parent_cache = dict(parent.parent_cache) if parent else {}
        self.parent_cache[g] = self

    def filter(self, graph):
        """Return a context restricted to a graph's dependencies."""
        rval = self.parent_cache.get(graph, None)
        if rval is None:
            rval = self.parent_cache.get(graph.parent, None)
        return rval

    def add(self, graph, argvals):
        """Extend this context with values for another graph."""
        parent = self.parent_cache.get(graph.parent, None)
        return Context(parent, graph, argvals)

    def __hash__(self):
        return hash((self.parent, self.graph, self.argkey))

    def __eq__(self, other):
        return type(other) is Context \
            and self.parent == other.parent \
            and self.graph == other.graph \
            and self.argkey == other.argkey

    async def __reify__(self):
        """Reify this Context."""
        return Context(
            await reify(self.parent),
            self.graph,
            await reify(self.argkey),
            True
        )


class Reference:
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
        g = node.value if is_constant_graph(node) else node.graph
        self.context = context.filter(g)

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

    def __eq__(self, other):
        return isinstance(other, Reference) \
            and self.node is other.node \
            and self.context == other.context

    def __hash__(self):
        return hash((self.node, self.context))


class VirtualReference:
    """Synthetic reference that can be given to an inferrer.

    A VirtualReference contains the values it is supposed to take on
    every track, so `engine.get(track, vr)` returns `vr.values[track]`.

    Attributes:
        values: The values for that reference on each track.

    """

    def __init__(self, values):
        """Initialize the VirtualReference."""
        self.values = values

    async def __getitem__(self, track):
        if track == '*':
            raise NotImplementedError('vref["*"]')
        else:
            return self.values[track]


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
                 eq_class=EquivalenceChecker):
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
        empty_context = Context.empty()
        root_context = empty_context.add(graph, argvals)
        output_ref = self.ref(graph.return_, root_context)

        async def _run():
            for track in self.required_tracks:
                inf = GraphInferrer(self.tracks[track],
                                    graph, empty_context)
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

        if isinstance(ref, VirtualReference):
            # A VirtualReference already contains the values we need.
            return await ref[track_name]

        node = ref.node
        inferred = ref.node.inferred.get(track_name, UNKNOWN)

        if inferred is not UNKNOWN:
            return inferred

        elif is_constant(node):
            return await track.infer_constant(ref)

        elif is_apply(node):
            return await track.infer_apply(ref)

        else:
            # Values for Parameters are cached when we enter a Graph.
            raise AssertionError(
                f'Cannot process: {node} in track "{track_name}"'
            )

    def get_inferred(self, track, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached.
        """
        return self.cache.get((track, ref))

    def run_coroutine(self, coro, throw=True):
        """Run an async function using this inferrer's loop."""
        errs_before = len(self.errors)
        try:
            fut = asyncio.ensure_future(coro, loop=self.loop)
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
