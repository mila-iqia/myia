
from .. import dtype, dshape
from ..infer import Track
from ..infer.utils import infer_trace
from ..ir import Graph
from ..prim import Primitive

from .base import from_vref


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
        v = ctref.node.value
        if isinstance(v, Graph):
            return GraphXInferrer(v, ctref.context)
        elif isinstance(v, Primitive):
            return self.constructors[v]()
        else:
            return from_vref(
                v,
                dtype.pytype_to_myiatype(type(v), v),
                dshape.NOSHAPE,
            )

    def from_value(self, v, context):
        return 8911

    def from_external(self, t):
        return 21

    def default(self, values):
        # return AbstractValue(values)
        # return AbstractTuple((1, 2, 3))
        return from_vref(
            values['value'],
            values['type'],
            values['shape'],
        )


class XInferrer:
    def __init__(self):
        self.cache = {}

    async def __call__(self, track, *refs):
        args = tuple([await ref['abstract'] for ref in refs])
        if args not in self.cache:
            self.cache[args] = await self.infer(track, *args)
        return self.cache[args]

    async def infer(self, track, *args):
        raise NotImplementedError()


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
            raise type_error_nargs(self.identifier, nargs, len(args))

        argkey, context = await self._make_argkey_and_context(track, args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(g.parameters, argkey):
            for track, v in arg:
                ref = engine.ref(p, context)
                engine.cache.set_value((track, ref), v)

        out = engine.ref(g.return_, context)
        return await engine.get_inferred('abstract', out)
