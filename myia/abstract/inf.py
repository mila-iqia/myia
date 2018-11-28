
from functools import reduce

from .. import dtype, dshape
from ..infer import Track, MyiaTypeError, Context
from ..infer.core import Pending
from ..infer.graph_infer import type_error_nargs, VirtualReference
from ..infer.utils import infer_trace
from ..ir import Graph
from ..prim import Primitive
from ..prim.py_implementations import typeof
from ..utils import as_frozen, Var, RestrictedVar, Overload, Partializable

from .base import from_vref, shapeof, AbstractScalar, Possibilities, \
    ABSENT, GraphAndContext, AbstractBase, amerge, bind, PartialApplication


_number_types = [
    dtype.Int[8], dtype.Int[16], dtype.Int[32], dtype.Int[64],
    dtype.UInt[8], dtype.UInt[16], dtype.UInt[32], dtype.UInt[64],
    dtype.Float[16], dtype.Float[32], dtype.Float[64],
]


class AbstractTrack(Track):
    def __init__(self,
                 engine,
                 name,
                 *,
                 constructors):
        super().__init__(engine, name)
        self.constructors = {
            prim: cons()
            for prim, cons in constructors.items()
        }
        self.subtracks = ['value', 'type', 'shape']

    get_inferrer_for = Overload()

    @get_inferrer_for.register
    def get_inferrer_for(self, prim: Primitive):
        return self.constructors[prim]

    @get_inferrer_for.register
    def get_inferrer_for(self, g: Graph):
        if g not in self.constructors:
            self.constructors[g] = GraphXInferrer(g, Context.empty())
        return self.constructors[g]

    @get_inferrer_for.register
    def get_inferrer_for(self, g: GraphAndContext):
        if g not in self.constructors:
            self.constructors[g] = GraphXInferrer(g.graph, g.context)
        return self.constructors[g]

    @get_inferrer_for.register
    def get_inferrer_for(self, part: PartialApplication):
        if part not in self.constructors:
            self.constructors[part] = PartialXInferrer(
                self.get_inferrer_for(part.fn),
                part.args
            )
        return self.constructors[part]

    async def infer_apply(self, ref):
        """Get the property for a ref of an Apply node."""
        ctx = ref.context
        n_fn, *n_args = ref.node.inputs
        # We await on the function node to get the inferrer
        fn_ref = self.engine.ref(n_fn, ctx)
        fn = await fn_ref[self.name]
        argrefs = [self.engine.ref(node, ctx) for node in n_args]

        infs = [self.get_inferrer_for(poss) for poss in fn.values['value']]

        return await self.engine.loop.schedule(
            execute_inferrers(self, infs, argrefs),
            context_map={
                infer_trace: {**infer_trace.get(), ctx: ref}
            }
        )

    async def infer_constant(self, ctref):
        """Get the property for a ref of a Constant node."""
        v = self.engine.pipeline.resources.convert(ctref.node.value)
        res = self.from_value(v, ctref.context)
        t = res.build('type')
        if dtype.ismyiatype(t, dtype.Number):
            prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            res.values['type'] = self.engine.loop.create_pending_from_list(
                _number_types, t, prio
            )
            # v = RestrictedVar(_number_types)
            # prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            # res.values['type'] = self.engine.loop.create_var(v, t, prio)
        return res

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return AbstractScalar({
                'value': Possibilities([v]),
                'type': dtype.Function,
                'shape': dshape.NOSHAPE,
            })
        elif isinstance(v, Graph):
            if v.parent:
                v = GraphAndContext(v, context)
            return AbstractScalar({
                'value': Possibilities([v]),
                'type': dtype.Function,
                'shape': dshape.NOSHAPE,
            })
        # elif isinstance(v, MetaGraph):
        #     return MetaGraphInferrer(self, v)
        # elif is_dataclass_type(v):
        #     rec = self.constructors[P.make_record]()
        #     typ = dtype.pytype_to_myiatype(v)
        #     vref = self.engine.vref({'value': typ, 'type': TypeType})
        #     return PartialInferrer(self, rec, [vref])
        else:
            return from_vref(v, typeof(v), shapeof(v))

    def from_external(self, t):
        return t

    def default(self, values):
        return from_vref(
            values['value'],
            values['type'],
            values['shape'],
        )

    # def abstract_merge(self, *values):
    #     resolved = []
    #     pending = set()
    #     committed = None
    #     for v in values:
    #         if isinstance(v, Pending):
    #             if v.resolved():
    #                 resolved.append(v.result())
    #             else:
    #                 pending.add(v)
    #         else:
    #             resolved.append(v)

    #     if pending:
    #         def resolve(fut):
    #             pending.remove(fut)
    #             result = fut.result()
    #             resolved.append(result)
    #             if not pending:
    #                 v = self.force_merge(resolved, model=committed)
    #                 rval.resolve_to(v)

    #         for p in pending:
    #             p.add_done_callback(resolve)

    #         def premature_resolve():
    #             nonlocal committed
    #             committed = self.force_merge(resolved)
    #             resolved.clear()
    #             return committed

    #         rval = self.engine.loop.create_pending(
    #             resolve=premature_resolve,
    #             priority=-1,
    #         )
    #         rval.equiv.update(values)
    #         for p in pending:
    #             p.tie(rval)
    #         return rval
    #     else:
    #         return self.force_merge(resolved)

    # def force_merge(self, values, model=None):
    #     if model is None:
    #         return reduce(self.merge, values)
    #     else:
    #         return reduce(self.accept, values)

    # def merge(self, x, y):
    #     if isinstance(x, AbstractBase):
    #         return x.merge(y)
    #     else:
    #         if x != y:
    #             raise MyiaTypeError(f'Cannot merge {x} and {y} (1)')
    #         return x

    # def accept(self, x, y):
    #     if isinstance(x, AbstractBase):
    #         return x.accept(y)
    #     else:
    #         if x != y:
    #             raise MyiaTypeError(f'Cannot merge {x} and {y} (2)')
    #         return x

    def abstract_merge(self, *values):
        return reduce(self._merge, values)

    def _merge(self, x1, x2):
        return amerge(x1, x2, loop=self.engine.loop, forced=False)

    def check_predicate(self, predicate, res):
        if isinstance(predicate, tuple):
            return any(self.check_predicate(p, res) for p in predicate)
        elif dtype.ismyiatype(predicate):
            return dtype.ismyiatype(res, predicate)
        elif callable(predicate):
            return predicate(res)
        else:
            raise ValueError(predicate)  # pragma: no cover

    def assert_predicate(self, predicate, res):
        if not self.check_predicate(predicate, res):
            raise MyiaTypeError(f'Expected {predicate}')

    def chk(self, predicate, *values):
        for value in values:
            if isinstance(value, Pending):
                value.add_done_callback(
                    lambda fut: self.assert_predicate(
                        predicate, fut.result()
                    )
                )
            else:
                self.assert_predicate(predicate, value)
        return self.abstract_merge(*values)


class XInferrer(Partializable):
    def __init__(self):
        self.cache = {}

    def build(self, field):
        return None

    async def __call__(self, track, *refs):
        args = tuple([await ref['abstract'] for ref in refs])
        if args not in self.cache:
            self.cache[args] = await self.infer(track, *args)
        return self.cache[args]

    async def infer(self, track, *args):
        raise NotImplementedError()

    def __repr__(self):
        return f'{type(self)}'


class GraphXInferrer(XInferrer):

    def __init__(self, graph, context, broaden=True):
        super().__init__()
        self._graph = graph
        self.broaden = broaden
        if context is None:
            self.context = Context.empty()
        else:
            self.context = context.filter(graph)
        assert self.context is not None

    async def make_graph(self, args):
        return self._graph

    async def make_context(self, track, args):
        _, ctx = await self._make_argkey_and_context(track, args)
        return ctx

    async def _make_argkey_and_context(self, track, args):
        engine = track.engine
        g = await self.make_graph(args)
        argvals = []
        for arg in args:
            argval = {}
            for track_name, track in engine.tracks.items():
                result = await engine.get_inferred(track_name, arg)
                if self.broaden and not g.flags.get('flatten_inference'):
                    result = track.broaden(result)
                argval[track_name] = result
            argvals.append(argval)

        # Update current context using the fetched properties.
        argkey = as_frozen(argvals)
        return argkey, self.context.add(g, argkey)

    async def __call__(self, track, *args):
        if args not in self.cache:
            self.cache[args] = await self.infer(track, *args)
        return self.cache[args]

    async def infer(self, track, *args):
        engine = track.engine
        g = await self.make_graph(args)
        nargs = len(g.parameters)

        if len(args) != nargs:
            raise type_error_nargs(self, nargs, len(args))

        argkey, context = await self._make_argkey_and_context(track, args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(g.parameters, argkey):
            for track, v in arg:
                ref = engine.ref(p, context)
                engine.cache.set_value((track, ref), v)

        out = engine.ref(g.return_, context)
        return await engine.get_inferred('abstract', out)


class PartialXInferrer(XInferrer):

    def __init__(self, fn, args):
        super().__init__()
        self.fn = fn
        self.args = args

    async def __call__(self, track, *refs):
        args = tuple([await ref['abstract'] for ref in refs])
        args = tuple(VirtualReference({'abstract': arg})
                     for arg in self.args + args)
        if args not in self.cache:
            self.cache[args] = await self.fn(track, *args)
        return self.cache[args]



async def _xinf_helper(track, inf, args, p):
    result = await inf(track, *args)
    p.resolve_to(result)


async def execute_inferrers(track, inferrers, args):
    if len(inferrers) == 1:
        inf, = inferrers
        return await inf(track, *args)

    else:
        pending = []
        for inf in inferrers:
            p = track.engine.loop.create_pending(
                resolve=None,
                priority=None
            )
            pending.append(p)
            track.engine.loop.schedule(
                _xinf_helper(track, inf, args, p)
            )

        return bind(track.engine.loop, None, [], pending)
