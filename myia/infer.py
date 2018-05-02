
import asyncio

from .dtype import Int, Bool, Float, Tuple, List
from .prim import ops as P, Primitive
from .ir import Graph, Constant, \
    is_constant, is_constant_graph, is_apply, is_parameter
from .utils import Named, Registry
from .cconv import NestingAnalyzer
from .prim.py_implementations import implementations as pyimpl


ANYTHING = Named('ANYTHING')


class MyiaTypeError(Exception):
    pass


def typeof(v):
    if isinstance(v, bool):
        return Bool()
    elif isinstance(v, int):
        return Int(64)
    elif isinstance(v, float):
        return Float(64)
    elif isinstance(v, tuple):
        return Tuple(map(typeof, v))
    else:
        raise TypeError(f'Untypable value: {v}')


####################
# Inferrer classes #
####################


class Inferrer:
    def __init__(self, engine):
        self.engine = engine
        self.cache = {}

    async def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = await self.infer(*args)
        return self.cache[args]


class PrimitiveInferrer(Inferrer):
    def __init__(self, engine, track, inferrer_fn):
        super().__init__(engine)
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
        super().__init__(engine)
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


def primitive_inferrer(track, prim):
    def wrap(fn):
        p = lambda engine: PrimitiveInferrer(engine, track, fn)
        all_inferrers[track].register(prim)(p)
        return p
    return wrap


class PrimitiveValueInferrer(Inferrer):
    def __init__(self, engine, impl):
        super().__init__(engine)
        self.impl = impl

    async def infer(self, *refs):
        coros = [self.engine.get('value', ref) for ref in refs]
        args = await asyncio.gather(*coros, loop=self.engine.loop)
        if any(arg is ANYTHING for arg in args):
            return ANYTHING
        else:
            return self.impl(*args)


#############
# Inferrers #
#############


all_inferrers = {
    'value': Registry(),
    'type': Registry()
}


async def infer_value_constant(engine, ct):
    v = ct.node.value
    if isinstance(v, Primitive):
        vinfs = all_inferrers['value']
        if v in vinfs:
            return vinfs[v](engine)
        else:
            return PrimitiveValueInferrer(engine, pyimpl[v])
    elif isinstance(v, Graph):
        return GraphInferrer(engine, 'value', v, ct.context)
    else:
        return v


@primitive_inferrer('value', P.if_)
async def infer_value_if(engine, cond, tb, fb):
    v = await engine.get('value', cond)
    if v is True:
        fn = await engine.get('value', tb)
    elif v is False:
        fn = await engine.get('value', fb)
    elif v is ANYTHING:
        return ANYTHING

    return await fn()


async def infer_type_constant(engine, ct):
    v = ct.node.value
    if isinstance(v, Primitive):
        return all_inferrers['type'][v](engine)
    elif isinstance(v, Graph):
        return GraphInferrer(engine, 'type', v, ct.context)
    else:
        return typeof(ct.node.value)


@primitive_inferrer('type', P.if_)
async def infer_type_if(engine, cond, tb, fb):
    assert await engine.get('type', cond) == Bool()
    v = await engine.get('value', cond)
    tb_res = (await engine.get('type', tb))()
    fb_res = (await engine.get('type', fb))()
    if v is True:
        return await tb_res
    elif v is False:
        return await fb_res
    elif v is ANYTHING:
        return await engine.force_same('type', tb_res, fb_res)


@primitive_inferrer('type', P.cons_tuple)
async def infer_type_cons_tuple(engine, x, y):
    x_t = await engine.get('type', x)
    y_t = await engine.get('type', y)
    assert isinstance(y_t, Tuple)
    return Tuple([x_t, *y_t.elements])


@primitive_inferrer('type', P.getitem)
async def infer_type_getitem(engine, seq, idx):
    seq_t = await engine.get('type', seq)
    idx_t = await engine.get('type', idx)
    if not isinstance(idx_t, Int):
        raise MyiaTypeError('Expected Int for index')

    if isinstance(seq_t, Tuple):
        idx_v = await engine.get('value', idx)
        assert idx_v is not ANYTHING
        return seq_t.elements[idx_v]
    elif isinstance(seq_t, List):
        return seq_t.element_type
    else:
        raise MyiaTypeError('Wrong seq type for getitem')


async def infer_type_compare_bin(engine, x, y):
    t = await engine.force_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return Bool()


async def infer_type_arith_bin(engine, x, y):
    t = await engine.force_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


for op in [P.add, P.sub, P.mul]:
    all_inferrers['type'].register(op)(
        lambda engine: PrimitiveInferrer(
            engine, 'type', infer_type_arith_bin
        )
    )

for op in [P.lt, P.gt, P.eq, P.le, P.ge]:
    all_inferrers['type'].register(op)(
        lambda engine: PrimitiveInferrer(
            engine, 'type', infer_type_compare_bin
        )
    )


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

    def __iter__(self):
        return iter(self.parts)

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

    def __hrepr__(self, H, hrepr):
        return hrepr({'node': self.node, 'context': self.context})


class VirtualReference:
    def __init__(self, **values):
        self.values = values


########
# Core #
########


class InferrerEquivalenceClass:

    def __init__(self, engine, members):
        self.engine = engine
        self.checked_members = set()
        self.pending_members = members
        self.results = {}

    def update(self, other):
        self.pending_members.update(other.checked_members)
        self.pending_members.update(other.pending_members)
        self.pending_members -= self.checked_members

    def __iter__(self):
        return iter(self.checked_members | self.pending_members)

    async def check(self):
        maybe_changes = False

        for inf in self.pending_members:
            maybe_changes = True
            for args, expected in self.results.items():
                v = await inf(*args)
                self.engine.equiv.declare_equivalent(expected, v)
        self.checked_members.update(self.pending_members)
        self.pending_members = set()

        all_keys = set()
        for m in self.checked_members:
            all_keys.update(m.cache.keys())

        to_check = all_keys - set(self.results.keys())

        for args in to_check:
            maybe_changes = True
            inf1, *others = self.checked_members
            res1 = inf1(*args)
            self.results[args] = res1
            for inf2 in others:
                res2 = inf2(*args)
                self.engine.equiv.declare_equivalent(res1, res2)

        return maybe_changes


class EquivalencePool:

    def __init__(self, engine):
        self.engine = engine
        self.eqclasses = {}
        self.pending = []
        self.checked = {}

    def declare_equivalent(self, x, y):
        self.pending.append((x, y))
        self.engine.schedule_function(self.check)

    async def _process_equivalence(self, x, y):
        vx = await x
        vy = await y

        if isinstance(vx, Inferrer) and isinstance(vy, Inferrer):
            if vx.provably_equivalent(vy):
                return

            eqx = self.eqclasses.get(
                vx, InferrerEquivalenceClass(self.engine, {vx})
            )
            eqy = self.eqclasses.get(
                vy, InferrerEquivalenceClass(self.engine, {vy})
            )
            eqx.update(eqy)
            for z in eqx:
                self.eqclasses[z] = eqx

        elif vx == vy:
            pass

        else:
            self.engine.errors.add(
                MyiaTypeError(f'Type mismatch: {vx} != {vy}')
            )

    async def check(self):
        maybe_changes = False
        pending, self.pending = self.pending, []
        for x, y in pending:
            await self._process_equivalence(x, y)
            maybe_changes = True

        for eq in set(self.eqclasses.values()):
            maybe_changes |= await eq.check()

        if maybe_changes:
            self.engine.schedule_function(self.check)


class EqEquivalence:
    def __init__(self, engine):
        self.engine = engine

    def declare_equivalent(self, x, y):
        self.engine.schedule(self.assert_equivalent(x, y))

    async def assert_equivalent(self, x, y):
        vx = await x
        vy = await y
        if not (vx == vy):
            self.engine.errors.add(TypeError(f'Type mismatch: {vx} != {vy}'))


class InferenceEngine:

    def __init__(self, constant_inferrers, eq_class=EquivalencePool):
        self.loop = asyncio.get_event_loop()
        self.tracks = tuple(constant_inferrers.keys())
        self.constant_inferrers = constant_inferrers
        self.cache = {track: {} for track in self.tracks}
        self.todo = set()
        self.errors = set()
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
            return await inf(*argrefs)
        else:
            raise Exception(f'Cannot process: {node}')

    def get(self, track, ref):
        futs = self.cache[track]
        if ref not in futs:
            futs[ref] = self.loop.create_task(self.compute_ref(track, ref))
        return futs[ref]

    def cache_value(self, track, ref, value):
        fut = asyncio.Future()
        fut.set_result(value)
        self.cache[track][ref] = fut

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
            todo = [t() for t in todo]
            await asyncio.gather(*todo, loop=self.loop)

        if self.errors:
            raise self.errors.pop()
        else:
            return {track: self.cache[track][oref].result()
                    for track in self.tracks}

    def run_sync(self, graph, args):
        self.deps = NestingAnalyzer(graph).graph_dependencies_total()
        return self.loop.run_until_complete(self.run(graph, args))

    async def force_same(self, track, *refs):
        futs = [self.get(track, ref) if isinstance(ref, Reference) else ref
                for ref in refs]
        done, pending = await asyncio.wait(
            futs,
            loop=self.loop,
            return_when=asyncio.FIRST_COMPLETED
        )
        main = done.pop()
        for fut in done|pending:
            self.equiv.declare_equivalent(fut, main)
        return main.result()
