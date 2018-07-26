"""Inference engine (types, values, etc.)."""

import asyncio
from heapq import heappush, heappop
from types import FunctionType

from .dtype import Type
from .ir import is_constant, is_constant_graph, is_apply
from .utils import Named, Partializable


# Represents an unknown value
ANYTHING = Named('ANYTHING')


class InferenceError(Exception):
    """Inference error in a Myia program.

    Attributes:
        message: The error message.
        refs: A list of references which are involved in the error,
            e.g. because they have the wrong type or don't match
            each other.
        traceback_refs: A map from a context to the first reference in
            that context that fails to resolve because of this error.
            This represents a traceback of sorts.

    """

    def __init__(self, message, refs=[]):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.traceback_refs = {}


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""

    pass


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f'Wrong number of arguments for {ident}:'
        f' expected {expected}, got {got}.'
    )


class ValueWrapper:
    """Wrapper for an inferred value.

    Values may be wrapped using subclasses of ValueWrapper, associating them
    with tracking data or metadata.
    """

    def __init__(self, value):
        """Initialize a ValueWrapper."""
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return type(other) is type(self) \
            and self.value == other.value

    @property
    def __unwrapped__(self):
        return self.value


#########
# Track #
#########


class Track(Partializable):
    """Represents a property to infer."""

    def __init__(self, engine, name):
        """Initialize a Track."""
        self.engine = engine
        self.name = name

    def infer_constant(self, ctref):
        """Get the property for a constant Reference."""
        return self.from_value(ctref.node.value, ctref.context)

    def from_value(self, v, context=None):
        """Get the property from a value in the context."""
        raise NotImplementedError()  # pragma: no cover

    def broaden(self, v):
        """Broaden the value for use in a graph's signature."""
        return v

    def default(self):
        """Default value for this track, if nothing is known."""
        raise NotImplementedError()  # pragma: no cover

    def fill_in(self, values, context=None):
        """Fill in a value for this property, given others."""
        if self.name in values:
            pass
        elif 'value' in values:
            v = self.engine.unwrap(values['value'])
            v = getattr(v, '__uninfer__', v)
            values[self.name] = self.from_value(v, context)
        else:
            values[self.name] = self.default()  # pragma: no cover

    def assert_same(self, *refs):
        """Assert that all refs have the same value on this track."""
        return self.engine.assert_same(self.name, *refs)

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

    async def expect(self,
                     predicate,
                     *refs,
                     assert_same=True):
        """Assert that all refs match the given predicate.

        Arguments:
            predicate: A type that all refs must have, or a predicate
                function, or a tuple of types/predicates.
            refs: The references to compare.
            assert_same: Whether all references should be identical
                on that track, in addition to matching the predicate.
        """
        results = [(ref, await ref[self.name]) for ref in refs]

        for ref, res in results:
            if not self.apply_predicate(predicate, res):
                raise self.predicate_error(
                    predicate,
                    res,
                    ref
                )

        if assert_same:
            (main_ref, main), *rest = results
            for ref, res in rest:
                if res != main:
                    raise InferenceError(
                        f"Mismatch: {main} != {res}",
                        [main_ref, ref]
                    )

        rval = [res for _, res in results]
        if len(rval) == 1:
            rval, = rval
        return rval


####################
# Inferrer classes #
####################


class Inferrer:
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
        self.track = track
        self.engine = track.engine
        self.identifier = identifier
        self.cache = {}

    async def __call__(self, *args):
        """Infer a property of the operation on the given arguments.

        The results of this call are cached.
        """
        if args not in self.cache:
            self.cache[args] = await self.infer(*args)
        return self.cache[args]

    def infer(self, *args):
        """Infer a property of the operation on the given arguments.

        This must be overriden in subclasses.
        """
        raise NotImplementedError()  # pragma: no cover

    def provably_equivalent(self, other):
        """Whether this inferrer is provably equivalent to the other."""
        return self is other  # pragma: no cover


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
            for track in self.engine.all_track_names:
                tr = self.engine.tracks[track]
                result = await self.engine.get_raw(track, arg)
                if not self.graph.flags.get('flatten_inference'):
                    result = tr.broaden(result)
                argval[track] = result
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
                self.engine.cache_value(track, ref, v)

        out = self.engine.ref(self.graph.return_, context)
        return await self.engine.get_raw(self.track.name, out)

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

    def __init__(self, parent, g, argvals):
        """Initialize the Context."""
        self.parent = parent
        self.graph = g
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

    def __getitem__(self, track):
        return self.engine.get(track, self)

    def get_raw(self, track):
        """Get the raw value for the track, which might be wrapped."""
        return self.engine.get_raw(track, self)

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

    def __init__(self, **values):
        """Initialize the VirtualReference."""
        self.values = values

    async def __getitem__(self, track):
        return self.values[track]


########
# Core #
########


class _TodoEntry:
    def __init__(self, order, handler):
        self.order = order
        self.handler = handler

    def __lt__(self, other):
        return self.order < other.order


class _InferenceLoop(asyncio.AbstractEventLoop):
    """EventLoop implementation for use with the inferrer.

    This event loop doesn't allow scheduling tasks and callbacks with
    `call_later` or `call_at`, which means the `timeout` argument to methods
    like `wait` will not work. `run_forever` will stop when it has exhausted
    all work there is to be done. This means `run_until_complete` may finish
    before it can evaluate the future, which suggests an infinite loop.
    """

    def __init__(self, debug=False):
        self._running = False
        self._todo = []
        self._debug = debug
        self._futures = []
        self._errors = []

    def get_debug(self):
        return self._debug

    def run_forever(self):
        self._running = True
        while self._todo and self._running:
            todo = heappop(self._todo)
            h = todo.handler
            if isinstance(h, asyncio.Handle):
                h._run()
            else:
                fut = asyncio.ensure_future(h, loop=self)
                self._futures.append(fut)

    def is_running(self):
        return self._running

    def is_closed(self):
        return not self.is_running()

    def schedule(self, x, order=0):
        heappush(self._todo, _TodoEntry(order, x))

    def collect_errors(self):
        futs, self._futures = self._futures, []
        errors, self._errors = self._errors, []
        for fut in futs:
            if fut.done():
                exc = fut.exception()
            else:
                exc = InferenceError(
                    f'Could not run inference to completion.'
                    ' There might be an infinite loop in the program'
                    ' which prevents type inference from working.',
                    refs=[]
                )
            if exc is not None:
                errors.append(exc)
        return errors

    def call_soon(self, callback, *args, context=None):
        h = asyncio.Handle(callback, args, self)
        heappush(self._todo, _TodoEntry(0, h))
        return h

    def call_later(self, delay, callback, *args, context=None):
        raise NotImplementedError(
            '_InferenceLoop does not allow timeouts or time-based scheduling.'
        )

    def call_at(self, when, callback, *args, context=None):
        raise NotImplementedError(
            '_InferenceLoop does not allow time-based scheduling.'
        )

    def create_task(self, coro):
        return asyncio.Task(coro, loop=self)

    def create_future(self):
        return asyncio.Future(loop=self)


class InferrerEquivalenceClass:
    """An equivalence class between a set of Inferrers.

    Two inferrers are equivalent if they return the same value on each set of
    arguments that they are given.
    """

    def __init__(self, engine, members):
        """Initialize the InferrerEquivalenceClass."""
        self.engine = engine
        self.all_members = members

        # Members for which we are up to date
        self.checked_members = set()

        # Members left to process
        self.pending_members = members

        # Map argument tuples to results
        self.results = {}

    def update(self, other):
        """Merge the other equivalence class into this one."""
        self.all_members.update(other.all_members)
        self.pending_members = self.all_members - self.checked_members

    def __iter__(self):
        return iter(self.all_members)

    async def check(self):
        """Check that all inferrers in the class are equivalent.

        Return True if any work was done (it might be necessary to run this
        method more than once, if new work entails that some members may be run
        on new arguments, or new members may be added to this equivalence
        class).
        """
        maybe_changes = False

        for inf in self.pending_members:
            maybe_changes = True
            # The assert will be triggered if we add new equivalences
            # after the first ones are resolved. I'm not certain in what
            # case exactly that'd happen. If it does, try to uncomment
            # the block after the assert and see if it works.
            assert not self.results, \
                "Congrats, you found out how to trigger this code."
            # for args, expected in self.results.items():
            #     v = inf(*args)
            #     self.engine.equiv.declare_equivalent(
            #         expected, v, self.all_members
            #     )
        self.checked_members.update(self.pending_members)
        self.pending_members = set()

        all_keys = set()
        for m in self.checked_members:
            all_keys.update(m.cache.keys())

        to_check = all_keys - set(self.results.keys())

        for args in to_check:
            maybe_changes = True
            inf1, *others = self.checked_members
            res1 = await inf1(*args)
            self.results[args] = res1
            for inf2 in others:
                res2 = inf2(*args)
                self.engine.equiv.declare_equivalent(
                    res1, res2, self.all_members
                )

        return maybe_changes


class EquivalencePool:
    """Handle equivalence between values.

    Equivalence between inferrers are handled with InferrerEquivalenceClass.
    """

    def __init__(self, engine):
        """Initialize the EquivalencePool."""
        self.engine = engine
        self.eqclasses = {}
        self.pending = []
        self.checked = {}

    def declare_equivalent(self, x, y, refs):
        """Declare that x and y should be equivalent.

        If an error occurs, the refs argument is to be packaged with it.
        """
        # We only put x and y in a pending list for now.
        self.pending.append((x, y, refs))
        # Later, we will call check.
        self.engine.schedule(self.check())

    async def _process_equivalence(self, x, y, refs):
        if hasattr(x, '__await__'):
            x = await x
        if hasattr(y, '__await__'):
            y = await y

        if isinstance(x, Inferrer) and isinstance(y, Inferrer):
            if x.provably_equivalent(y):
                return

            # We merge the equivalence classes for x and y.
            eqx = self.eqclasses.get(
                x, InferrerEquivalenceClass(self.engine, {x})
            )
            eqy = self.eqclasses.get(
                y, InferrerEquivalenceClass(self.engine, {y})
            )
            eqx.update(eqy)
            for z in eqx:
                self.eqclasses[z] = eqx

        elif x == y:
            pass

        else:
            # We log the error directly instead of raising an exception so that
            # it doesn't prevent other (independent) checks.
            self.engine.errors.append(
                MyiaTypeError(f'Type mismatch: {x} != {y}', refs=refs)
            )

    async def check(self):
        """Check whether the declared equivalences hold.

        If new work was performed in order to perform the checks, `check`
        reschedules itself on the engine, in case there are new equivalences,
        or equivalences to rescind.
        """
        maybe_changes = False

        # Process all pending tasks
        pending, self.pending = self.pending, []
        for x, y, refs in pending:
            await self._process_equivalence(x, y, refs)
            maybe_changes = True

        # Process the inferrers using equivalence classes
        for eq in set(self.eqclasses.values()):
            maybe_changes |= await eq.check()

        if maybe_changes:
            # Reschedule, in case there is new data.
            self.engine.schedule(self.check(), order=1000)


class InferenceEngine:
    """Infer various properties about nodes in graphs.

    Arguments:
        graph: The graph to analyze.
        argvals: The arguments. Must be a tuple of dictionaries where
            each dictionary maps track name to value.
        tracks: Map each track (property name) to a Track object.
        required_tracks: A list of tracks that will be inferred for
            the output. Other tracks may be used if requested in
            the evaluation of a required track.
        eq_class: The class to use to check equivalence between
            values.

    """

    def __init__(self,
                 pipeline,
                 graph,
                 argvals,
                 *,
                 tracks,
                 required_tracks=None,
                 eq_class=EquivalencePool):
        """Initialize the InferenceEngine."""
        self.pipeline = pipeline

        self.graph = graph

        self.all_track_names = tuple(tracks.keys())
        self.tracks = {
            name: t(engine=self, name=name)
            for name, t in tracks.items()
        }
        self.required_tracks = required_tracks or self.all_track_names

        self.argrefs = [self.vref(arg) for arg in argvals]
        self.argvals = [ref.values for ref in self.argrefs]

        self.cache = {track: {} for track in self.all_track_names}
        self.errors = []
        self.equiv = eq_class(self)

        self.mng = self.pipeline.resources.manager
        self.mng.add_graph(graph)
        empty_context = Context.empty()
        self.root_context = empty_context.add(graph, argvals)

        self.loop = _InferenceLoop()
        self.run_coroutine(self._run(), throw=False)

    def ref(self, node, context):
        """Return a Reference to the node in the given context."""
        return Reference(self, node, context)

    def vref(self, values):
        """Return a VirtualReference using the given property values."""
        for track_obj in self.tracks.values():
            track_obj.fill_in(values)
        return VirtualReference(**values)

    async def compute_ref(self, track, ref):
        """Compute the value of the Reference on the given track."""
        if isinstance(ref, VirtualReference):
            # A VirtualReference already contains the values we need.
            return await ref[track]

        node = ref.node
        ctx = ref.context

        if is_constant(node):
            return self.tracks[track].infer_constant(ref)

        elif is_apply(node):
            n_fn, *n_args = node.inputs
            # We await on the function node to get the inferrer
            inf = await self.get(track, self.ref(n_fn, ctx))
            if inf is ANYTHING:
                return ANYTHING
            if not isinstance(inf, Inferrer):
                raise AssertionError(f'Not an inferrer: {inf}')
            argrefs = [self.ref(node, ctx) for node in n_args]
            try:
                return await inf(*argrefs)
            except RuntimeError as e:
                # At some point, some invalid recursive graphs raised this
                # error. You can just add a log_error if this happens again
                # e.g. because of changes in the inference engine.
                raise  # pragma: no cover
            except InferenceError as infe:
                # This builds a traceback of sorts in traceback_refs
                # The first encounter with a ctx will be the caller,
                # the others will be subsequence operations that depend
                # on the result.
                infe.traceback_refs.setdefault(ctx, ref)
                raise

        else:
            # Values for Parameters are cached when we enter a Graph.
            raise Exception(f'Cannot process: {node}')  # pragma: no cover

    def get_raw(self, track, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached. This method may return a wrapper around the
        desired value, depending on the track.
        """
        futs = self.cache[track]
        if ref not in futs:
            if self.loop.is_closed():
                raise Exception('Requested an unprocessed reference.') \
                    # pragma: no cover
            futs[ref] = self.loop.create_task(self.compute_ref(track, ref))
        return futs[ref]

    def unwrap(self, v):
        """Unwrap a cached value."""
        return getattr(v, '__unwrapped__', v)

    async def get(self, track, ref):
        """Get a Future for the value of the Reference on the given track.

        Results are cached.
        """
        v = await self.get_raw(track, ref)
        return self.unwrap(v)

    def get_info(self, track, ref):
        """Get information on the given track for the given reference.

        This assumes that the Future associated to the information has
        already been resolved. Asynchronous Inferrers should use the get
        method instead.
        """
        v = self.get_raw(track, ref).result()
        return self.unwrap(v)

    def output_info(self):
        """Return information about the output of the analyzed graph."""
        if self.errors:
            raise self.errors[0]  # pragma: no cover
        else:
            oref = self.ref(self.graph.return_, self.root_context)
            return {track: self.get_info(track, oref)
                    for track in self.required_tracks}

    def cache_value(self, track, ref, value):
        """Set the value of the Reference on the given track."""
        # We create a future and resolve it immediately, because all entries
        # in the cache must be futures.
        fut = asyncio.Future(loop=self.loop)
        fut.set_result(value)
        self.cache[track][ref] = fut

    def schedule(self, coro, order=0):
        """Schedule the given coroutine to run."""
        self.loop.schedule(coro, order)

    async def _run(self):
        for track in self.required_tracks:
            inf = GraphInferrer(self.tracks[track],
                                self.graph, self.root_context)
            self.schedule(inf(*self.argrefs))

    async def assert_same(self, track, *refs):
        """Assert that all refs have the same value on the given track."""
        # Make a future for the value of each reference
        futs = [self.get(track, ref) if isinstance(ref, Reference) else ref
                for ref in refs]

        # We wait only for the first future to complete
        done, pending = await asyncio.wait(
            futs,
            loop=self.loop,
            return_when=asyncio.FIRST_COMPLETED
        )

        # We must now tell equiv that all remaining futures must return the
        # same thing as the first one. This will essentially schedule a
        # bunch of tasks to wait for the remaining futures and verify that
        # they match. See EquivalencePool.
        main = done.pop()
        for fut in done | pending:
            self.equiv.declare_equivalent(fut, main, refs)

        # We return the first result immediately
        return main.result()

    def run_coroutine(self, coro, throw=True):
        """Run an async function using this inferrer's loop."""
        errs_before = len(self.errors)
        try:
            fut = asyncio.ensure_future(coro, loop=self.loop)
            self.loop.run_forever()
            self.errors.extend(self.loop.collect_errors())
            if errs_before < len(self.errors):
                if throw:  # pragma: no cover
                    raise self.errors[-1]
                else:
                    return None
            return fut.result()
        finally:
            for task in asyncio.Task.all_tasks(self.loop):
                task._log_destroy_pending = False
