
import asyncio

from .ir import \
    is_constant, is_constant_graph, is_apply
from .utils import Named
from .cconv import NestingAnalyzer


ANYTHING = Named('ANYTHING')


class MyiaTypeError(Exception):
    pass


####################
# Inferrer classes #
####################


class Inferrer:
    def __init__(self, engine, identifier):
        self.engine = engine
        self.identifier = identifier
        self.cache = {}

    async def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = await self.infer(*args)
        return self.cache[args]


class PrimitiveInferrer(Inferrer):
    def __init__(self, engine, track, prim, inferrer_fn):
        super().__init__(engine, prim)
        self.track = track
        self.inferrer_fn = inferrer_fn

    def infer(self, *args):
        return self.inferrer_fn(self.engine, *args)

    def provably_equivalent(self, other):
        return isinstance(other, PrimitiveInferrer) \
            and other.track == self.track \
            and other.inferrer_fn == self.inferrer_fn


class GraphInferrer(Inferrer):
    def __init__(self, engine, track, graph, context):
        super().__init__(engine, graph)
        self.track = track
        self.graph = graph
        self.context = context.filter(graph)

    async def infer(self, *args):
        engine = self.engine
        argvals = {
            track: await asyncio.gather(
                *[engine.get(track, arg) for arg in args],
                loop=engine.loop
            )
            for track in engine.tracks
        }

        gsigs = {(self.graph, track, tuple(argvals[track]))
                 for track in engine.tracks}
        context = self.context | gsigs

        for track, vals in argvals.items():
            for p, v in zip(self.graph.parameters, vals):
                ref = Reference(p, context)
                engine.cache_value(track, ref, v)

        out = Reference(self.graph.output, context)
        return await engine.get(self.track, out)

    def provably_equivalent(self, other):
        return isinstance(other, GraphInferrer) \
            and other.track == self.track \
            and other.graph == self.graph \
            and other.context == self.context


##############
# References #
##############


class Context:
    def __init__(self, engine, parts):
        self.engine = engine
        self.parts = frozenset(parts)

    def filter(self, graph):
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
    def __init__(self, node, context):
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
    def __init__(self, **values):
        self.values = values


########
# Core #
########


class InferrerEquivalenceClass:

    def __init__(self, engine, members):
        self.engine = engine
        self.all_members = members
        self.checked_members = set()
        self.pending_members = members
        self.results = {}

    def update(self, other):
        self.all_members.update(other.all_members)
        self.pending_members = self.all_members - self.checked_members

    def __iter__(self):
        return iter(self.all_members)

    async def check(self):
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

    def __init__(self, engine):
        self.engine = engine
        self.eqclasses = {}
        self.pending = []
        self.checked = {}

    def declare_equivalent(self, x, y, refs):
        self.pending.append((x, y, refs))
        self.engine.schedule_function(self.check)

    async def _process_equivalence(self, x, y, refs):
        if hasattr(x, '__await__'):
            x = await x
        if hasattr(y, '__await__'):
            y = await y

        if isinstance(x, Inferrer) and isinstance(y, Inferrer):
            if x.provably_equivalent(y):
                return

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
        maybe_changes = False
        pending, self.pending = self.pending, []
        for x, y, refs in pending:
            await self._process_equivalence(x, y, refs)
            maybe_changes = True

        for eq in set(self.eqclasses.values()):
            maybe_changes |= await eq.check()

        if maybe_changes:
            self.engine.schedule_function(self.check)


class InferenceEngine:

    def __init__(self,
                 constant_inferrers,
                 eq_class=EquivalencePool,
                 timeout=1.0):
        self.loop = asyncio.new_event_loop()
        self.tracks = tuple(constant_inferrers.keys())
        self.constant_inferrers = constant_inferrers
        self.timeout = timeout
        self.cache = {track: {} for track in self.tracks}
        self.todo = set()
        self.errors = []
        self.equiv = eq_class(self)

    async def compute_ref(self, track, ref):
        if isinstance(ref, VirtualReference):
            return ref.values[track]
        node = ref.node
        ctx = ref.context
        if is_constant(node):
            return await self.constant_inferrers[track](self, ref)
        elif is_apply(node):
            n_fn, *n_args = node.inputs
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
                message = e.args[0]
                if message.startswith('Task cannot await on itself'):
                    e2 = self.log_error(
                        [ref], 'There seems to be an infinite recursion'
                    )
                    raise e2
                else:
                    raise  # pragma: no cover
        else:
            raise Exception(f'Cannot process: {node}')  # pragma: no cover

    def get(self, track, ref):
        futs = self.cache[track]
        if ref not in futs:
            futs[ref] = self.loop.create_task(self.compute_ref(track, ref))
        return futs[ref]

    def cache_value(self, track, ref, value):
        fut = asyncio.Future()
        fut.set_result(value)
        self.cache[track][ref] = fut

    def log_error(self, refs, err):
        if isinstance(err, str):
            err = MyiaTypeError(err)
        elif isinstance(err, asyncio.Future):
            err = err.exception()
            if err is None:
                return
        self.errors.append({'refs': refs, 'error': err})
        return err

    def schedule(self, coro):
        self.todo.add(lambda: coro)

    def schedule_function(self, fn):
        self.todo.add(fn)

    async def run(self, graph, args):
        assert len(graph.parameters) == len(args)

        ctx = Context(
            self,
            ((graph, track, tuple(arg[track] for arg in args))
             for track in self.tracks)
        )

        for p, arg in zip(graph.parameters, args):
            for track, value in arg.items():
                ref = Reference(p, ctx)
                self.cache_value(track, ref, value)

        oref = Reference(graph.output, ctx)

        for track in self.tracks:
            self.schedule(self.get(track, oref))

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

    def run_sync(self, graph, args):
        self.deps = NestingAnalyzer(graph).graph_dependencies_total()
        try:
            res = self.loop.run_until_complete(self.run(graph, args))
        finally:
            for task in asyncio.Task.all_tasks(self.loop):
                task._log_destroy_pending = False
            self.loop.close()
        return res

    async def force_same(self, track, *refs):
        futs = [self.get(track, ref) if isinstance(ref, Reference) else ref
                for ref in refs]
        done, pending = await asyncio.wait(
            futs,
            loop=self.loop,
            return_when=asyncio.FIRST_COMPLETED
        )
        for fut in done:
            self.log_error(refs, fut)
        main = done.pop()
        for fut in done | pending:
            self.equiv.declare_equivalent(fut, main, refs)
        return main.result()
