"""Inference engine (types, values, etc.)."""

import asyncio
from typing import Set, Tuple, Any

from .ir import \
    Graph, is_constant, is_constant_graph, is_apply
from .utils import Named
from .cconv import NestingAnalyzer


# Represents an unknown value
ANYTHING = Named('ANYTHING')


class MyiaTypeError(Exception):
    """Type error in a Myia program."""

    pass


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f'Wrong number of arguments for {ident}:'
        f' expected {expected}, got {got}.'
    )


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

    async def infer(self, *args):
        """Infer a property of the operation on the given arguments."""
        if len(args) != self.nargs:
            raise type_error_nargs(self.identifier, self.nargs, len(args))

        engine = self.engine

        # We fetch all relevant properties of all arguments in order to build a
        # context (this cannot be done lazily, like with primitives, because we
        # need a concrete context.).
        argvals = {
            track: await asyncio.gather(
                *[engine.get(track, arg) for arg in args],
                loop=engine.loop
            )
            for track in engine.tracks
        }

        # Update current context using the fetched properties.
        gsigs = {(self.graph, track, tuple(argvals[track]))
                 for track in engine.tracks}
        context = self.context | gsigs

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for track, vals in argvals.items():
            for p, v in zip(self.graph.parameters, vals):
                ref = Reference(p, context)
                engine.cache_value(track, ref, v)

        out = Reference(self.graph.output, context)
        return await engine.get(self.track, out)

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

    def __init__(self, engine, parts):
        """Initialize the Context."""
        self.engine = engine
        # parts = set of (graph, track/property_name, (*args))
        self.parts: Set[Tuple[Graph, str, Tuple[Any, ...]]] = \
            frozenset(parts)

    def filter(self, graph):
        """Return a context restricted to a graph's dependencies."""
        if graph is None:
            return Context(self.engine, ())
        deps = self.engine.deps[graph]
        return Context(
            self.engine,
            ((g, track, argsig)
             for (g, track, argsig) in self.parts
             if g in deps or g is graph)
        )

    def __hash__(self):
        return hash((self.engine, self.parts))

    def __eq__(self, other):
        return type(other) is Context \
            and self.engine == other.engine \
            and self.parts == other.parts

    def __or__(self, other):
        return Context(self.engine, self.parts | other)


class Reference:
    """Reference to a certain node in a certain context.

    Attributes:
        node: The ANFNode being referred to.
        context: The Context for the node.

    """

    def __init__(self, node, context):
        """Initialize the Reference."""
        self.node = node
        g = node.value if is_constant_graph(node) else node.graph
        self.context = context.filter(g)

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

    Attributes:
        constant_inferrers: Map each track (property name) to an
            async function that can infer that property on a
            Constant. Applying a constant_inferrer on a Constant
            that contains a Primitive or a Graph should return
            an Inferrer instance.
        eq_class: The class to use to check equivalence between
            values.
        timeout: Timeout applied when awaiting in the main loop,
            to check for possible deadlocks. This is not a hard
            limit on how long the inference might take.

    """

    def __init__(self,
                 constant_inferrers,
                 eq_class=EquivalencePool,
                 timeout=1.0):
        """Initialize the InferenceEngine."""
        self.loop = asyncio.new_event_loop()
        self.tracks = tuple(constant_inferrers.keys())
        self.constant_inferrers = constant_inferrers
        self.timeout = timeout
        self.cache = {track: {} for track in self.tracks}
        self.todo = set()
        self.errors = []
        self.equiv = eq_class(self)

    async def compute_ref(self, track, ref):
        """Compute the value of the Reference on the given track."""
        if isinstance(ref, VirtualReference):
            # A VirtualReference already contains the values we need.
            return ref.values[track]

        node = ref.node
        ctx = ref.context

        if is_constant(node):
            return await self.constant_inferrers[track](self, ref)

        elif is_apply(node):
            n_fn, *n_args = node.inputs
            # We await on the function node to get the inferrer
            inf = await self.get(track, Reference(n_fn, ctx))
            if inf is ANYTHING:
                return ANYTHING
            assert isinstance(inf, Inferrer)
            argrefs = [Reference(node, ctx) for node in n_args]
            try:
                return await inf(*argrefs)
            except MyiaTypeError as e:
                self.log_error([ref], e)
                raise
            except RuntimeError as e:
                # This happens sometimes, e.g. in
                # def f(x): return f(x - 1)
                message = e.args[0]
                if message.startswith('Task cannot await on itself'):
                    e2 = self.log_error(
                        [ref], 'There seems to be an infinite recursion.'
                    )
                    raise e2
                else:
                    raise  # pragma: no cover

        else:
            # Values for Parameters are cached when we enter a Graph.
            raise Exception(f'Cannot process: {node}')  # pragma: no cover

    def get(self, track, ref):
        """Get the value of the Reference on the given track.

        Results are cached.
        """
        futs = self.cache[track]
        if ref not in futs:
            futs[ref] = self.loop.create_task(self.compute_ref(track, ref))
        return futs[ref]

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
                * A string to wrap as a MyiaTypeError.
                * A future. If the future failed due to an exception,
                  the exception is logged, otherwise the method returns
                  without doing anything.
        """
        if isinstance(err, str):
            err = MyiaTypeError(err)
        elif isinstance(err, asyncio.Future):
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

    async def _run(self, graph, args):

        ctx = Context(
            self,
            ((graph, track, tuple(arg[track] for arg in args))
             for track in self.tracks)
        )
        refs = [VirtualReference(**arg) for arg in args]
        oref = Reference(graph.output, ctx)

        for track in self.tracks:
            inf = GraphInferrer(self, track, graph, ctx)
            self.schedule(inf(*refs))

        while self.todo:
            todo = list(self.todo)
            self.todo = set()
            todo = [t() if callable(t) else t for t in todo]
            done, pending = await asyncio.wait(
                todo, loop=self.loop, timeout=self.timeout
            )
            if not done:
                self.log_error(
                    [], MyiaTypeError(
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

        if self.errors:
            raise self.errors[0]['error']
        else:
            return {track: self.cache[track][oref].result()
                    for track in self.tracks}

    def run(self, graph, args):
        """Run inference on the given graph and arguments.

        Arguments:
            graph: The graph to process.
            args: The arguments. Must be a tuple of dictionaries where
                each dictionary maps track name to value.
        """
        self.deps = NestingAnalyzer(graph).graph_dependencies_total()
        try:
            res = self.loop.run_until_complete(self._run(graph, args))
        finally:
            for task in asyncio.Task.all_tasks(self.loop):
                task._log_destroy_pending = False
            self.loop.close()
        return res

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
