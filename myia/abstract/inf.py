
from .. import dtype, dshape
from ..infer import Track, MyiaTypeError, Context
from ..infer.graph_infer import type_error_nargs
from ..infer.utils import infer_trace
from ..ir import Graph
from ..prim import Primitive
from ..prim.py_implementations import typeof
from ..utils import as_frozen, Var, RestrictedVar, Overload, Partializable

from .base import from_vref, shapeof, AbstractScalar, Possibilities, \
    ABSENT, GraphAndContext


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
            v = RestrictedVar(_number_types)
            prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            res.values['type'] = self.engine.loop.create_var(v, t, prio)
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


async def execute_inferrers(track, inferrers, args):
    if len(inferrers) == 1:
        inf, = inferrers
        return await inf(track, *args)

    else:
        assert False
