
from .. import dtype, dshape
from ..infer import Track, MyiaTypeError
from ..infer.graph_infer import type_error_nargs
from ..infer.utils import infer_trace
from ..ir import Graph
from ..prim import Primitive
from ..prim.py_implementations import typeof
from ..utils import as_frozen, Var, RestrictedVar

from .base import from_vref, shapeof, AbstractValue


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
        self.constructors = constructors
        self.subtracks = ['value', 'type', 'shape']

    async def infer_apply(self, ref):
        """Get the property for a ref of an Apply node."""
        ctx = ref.context
        n_fn, *n_args = ref.node.inputs
        # We await on the function node to get the inferrer
        fn_ref = self.engine.ref(n_fn, ctx)
        inf = await fn_ref[self.name]
        argrefs = [self.engine.ref(node, ctx) for node in n_args]
        if not isinstance(inf, XInferrer):
            raise MyiaTypeError(
                f'Trying to call a non-callable type: {inf}',
                refs=[fn_ref],
                app=ref
            )
        return await self.engine.loop.schedule(
            inf(self, *argrefs),
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

    # async def infer_constant(self, ctref):
    #     v = ctref.node.value
    #     if isinstance(v, Graph):
    #         return GraphXInferrer(v, ctref.context)
    #     elif isinstance(v, Primitive):
    #         return self.constructors[v]()
    #     else:
    #         return from_vref(
    #             v,
    #             dtype.pytype_to_myiatype(type(v), v),
    #             dshape.NOSHAPE,
    #         )

    # def from_value(self, v, context):
    #     return 8911

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v]()
        elif isinstance(v, Graph):
            return GraphXInferrer(v, context)
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
        # return AbstractValue(values)
        # return AbstractTuple((1, 2, 3))
        return from_vref(
            values['value'],
            values['type'],
            values['shape'],
        )


class XInferrer(AbstractValue):
    def __init__(self, values={}):
        self.cache = {}
        super().__init__(values)

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
        super().__init__({'value': graph, 'type': None, 'shape': None})
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
