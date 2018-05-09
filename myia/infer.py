"""Inference engine (types, values, etc.)."""

import asyncio

from .ir import is_constant, is_constant_graph, is_apply
from .utils import Named
from .cconv import NestingAnalyzer


# Represents an unknown value
ANYTHING = Named('ANYTHING')


class InferenceError(Exception):
    """Inference error in a Myia program."""

    pass


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""

    pass


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f'Wrong number of arguments for {ident}:'
        f' expected {expected}, got {got}.'
    )


#########
# Track #
#########


class Track:
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

    def __init__(self, engine, identifier):
        """Initialize the Inferrer."""
        self.engine = engine
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

    def __init__(self, engine, prim, nargs, inferrer_fn):
        """Initialize the PrimitiveInferrer."""
        super().__init__(engine, prim)
        self.inferrer_fn = inferrer_fn
        self.nargs = nargs

    def infer(self, *args):
        """Infer a property of the operation on the given arguments."""
        if self.nargs is not None and len(args) != self.nargs:
            raise type_error_nargs(self.identifier, self.nargs, len(args))
        return self.inferrer_fn(self.engine, *args)

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

    def __init__(self, engine, track, graph, context):
        """Initialize the GraphInferrer."""
        super().__init__(engine, graph)
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
                argval[track] = tr.broaden(result)
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
        return await self.engine.get_raw(self.track, out)

    def provably_equivalent(self, other):
        """Whether this inferrer is provably equivalent to the other.

        Two GraphInferrers are equivalent if they infer the same property
        on the same graph in the same context.
        """
        return isinstance(other, GraphInferrer) \
            and other.track == self.track \
            and other.graph == self.graph \
            and other.context == self.context


def register_inferrer(*prims, nargs, constructors):
    """Define a PrimitiveInferrer for prims with nargs arguments.

    For each primitive, this registers a constructor for a PrimitiveInferrer
    that takes an engine argument, in the constructors dictionary.
    """
    def deco(fn):
        def make_constructor(prim):
            def constructor(engine):
                return PrimitiveInferrer(engine, prim, nargs, fn)
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
    def empty(cls, parents_map):
        """Create an empty context."""
        return Context(None, None, (), parents_map=parents_map)

    def __init__(self, parent, g, argvals, *, parents_map):
        """Initialize the Context."""
        self.parents_map = parents_map
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
            parent_graph = self.parents_map.get(graph, None)
            rval = self.parent_cache.get(parent_graph, None)
        return rval

    def add(self, graph, argvals):
        """Extend this context with values for another graph."""
        parent_graph = self.parents_map[graph]
        parent = self.parent_cache.get(parent_graph, None)
        return Context(parent, graph, argvals,
                       parents_map=self.parents_map)

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

    async def get_all(self):
        """Return all properties associated to this reference."""
        return {track: await self[track]
                for track in self.engine.all_track_names}

    def __getitem__(self, track):
        return self.engine.get(track, self)

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

    async def get_all(self):
        """Return all properties associated to this reference."""
        return self.values


########
# Core #
########


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
        self.engine.schedule_function(self.check)

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
            self.engine.log_error(
                refs, MyiaTypeError(f'Type mismatch: {x} != {y}')
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
            self.engine.schedule_function(self.check)


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
        timeout: Timeout applied when awaiting in the main loop,
            to check for possible deadlocks. This is not a hard
            limit on how long the inference might take.

    """

    def __init__(self,
                 graph,
                 argvals,
                 *,
                 tracks,
                 required_tracks=None,
                 eq_class=EquivalencePool,
                 timeout=1.0):
        """Initialize the InferenceEngine."""
        self.graph = graph

        self.all_track_names = tuple(tracks.keys())
        self.tracks = {
            name: t(self, name)
            for name, t in tracks.items()
        }
        self.required_tracks = required_tracks or self.all_track_names

        self.argrefs = [self.vref(arg) for arg in argvals]
        self.argvals = [ref.values for ref in self.argrefs]

        self.timeout = timeout
        self.cache = {track: {} for track in self.all_track_names}
        self.todo = set()
        self.errors = []
        self.equiv = eq_class(self)

        self.parents = NestingAnalyzer(graph).parents()
        empty_context = Context.empty(self.parents)
        self.root_context = empty_context.add(graph, argvals)

        self.loop = asyncio.new_event_loop()
        self.run_coroutine(self._run())

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
            assert isinstance(inf, Inferrer)
            argrefs = [self.ref(node, ctx) for node in n_args]
            try:
                return await inf(*argrefs)
            except InferenceError as e:
                self.log_error([ref], e)
                raise
            except RuntimeError as e:
                # At some point, some invalid recursive graphs raised this
                # error. You can just add a log_error if this happens again
                # e.g. because of changes in the inference engine.
                raise  # pragma: no cover

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

    def output_info(self, track):
        """Return information about the output of the analyzed graph."""
        if self.errors:
            raise self.errors[0]['error']
        else:
            oref = self.ref(self.graph.return_, self.root_context)
            return self.get_info(track, oref)

    def cache_value(self, track, ref, value):
        """Set the value of the Reference on the given track."""
        # We create a future and resolve it immediately, because all entries
        # in the cache must be futures.
        fut = asyncio.Future()
        fut.set_result(value)
        self.cache[track][ref] = fut

    def log_error(self, refs, err):
        """Log an error, with a context given by the given refs.

        Arguments:
            refs: A list of objects that give a context to the error,
                e.g. References.
            err: Can be one of:
                * An exception.
                * A string to wrap as an InferenceError.
                * A future. If the future failed due to an exception,
                  the exception is logged, otherwise the method returns
                  without doing anything.
        """
        if isinstance(err, asyncio.Future):
            err = err.exception()
            if err is None:
                return
        self.errors.append({'refs': refs, 'error': err})
        return err

    def schedule(self, coro):
        """Schedule the given coroutine to run."""
        self.todo.add(lambda: coro)

    def schedule_function(self, fn):
        """Schedule a function that returns a coroutine to run.

        Scheduling the same function multiple times before it runs will
        only cause it to run once. It can be rescheduled after a run.

        Arguments:
            fn: A nullary function that returns a coroutine to run.
        """
        self.todo.add(fn)

    async def _run(self):
        for track in self.required_tracks:
            inf = GraphInferrer(self, track,
                                self.graph, self.root_context)
            self.schedule(inf(*self.argrefs))

        while self.todo:
            todo = list(self.todo)
            self.todo = set()
            todo = [t() if callable(t) else t for t in todo]
            done, pending = await asyncio.wait(
                todo, loop=self.loop, timeout=self.timeout
            )
            if not done:
                self.log_error(
                    [], InferenceError(
                        f'Exceeded timeout ({self.timeout}s) in type inferrer.'
                        ' There might be an infinite loop in the program,'
                        ' or the program is too large and you should'
                        ' increase the timeout.'
                    )
                )
                break
            for d in done:
                self.log_error([], d)
            self.todo.update(pending)

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

        # Log any errors in the futures that finished
        for fut in done:
            self.log_error(refs, fut)

        # We must now tell equiv that all remaining futures must return the
        # same thing as the first one. This will essentially schedule a
        # bunch of tasks to wait for the remaining futures and verify that
        # they match. See EquivalencePool.
        main = done.pop()
        for fut in done | pending:
            self.equiv.declare_equivalent(fut, main, refs)

        # We return the first result immediately
        return main.result()

    def run_coroutine(self, coro):
        """Run an async function using this inferrer's loop."""
        try:
            res = self.loop.run_until_complete(coro)
        finally:
            for task in asyncio.Task.all_tasks(self.loop):
                task._log_destroy_pending = False
        return res

    def close(self):
        """Close this inferrer's loop."""
        self.loop.close()
