
import numpy as np
from functools import reduce
from dataclasses import is_dataclass

from .. import dtype
from ..infer import Track, MyiaTypeError, Context, ANYTHING
from ..infer.core import Pending, reify
from ..infer.graph_infer import type_error_nargs, VirtualReference
from ..infer.utils import infer_trace
from ..ir import Graph, MetaGraph, GraphGenerationError
from ..prim import Primitive, ops as P
from ..prim.py_implementations import typeof
from ..utils import as_frozen, Var, RestrictedVar, Overload, Partializable, \
    is_dataclass_type

from .base import AbstractScalar, Possibilities, \
    ABSENT, GraphAndContext, AbstractBase, amerge, bind, PartialApplication, \
    JTransformedFunction, AbstractJTagged, AbstractTuple, \
    sensitivity_transform, VirtualFunction, AbstractFunction, \
    VALUE, TYPE, SHAPE, DummyFunction, TrackableFunction, \
    TypedPrimitive, AbstractType, AbstractClass, AbstractArray, \
    AbstractList, broaden as _broaden


_number_types = [
    dtype.Int[8], dtype.Int[16], dtype.Int[32], dtype.Int[64],
    dtype.UInt[8], dtype.UInt[16], dtype.UInt[32], dtype.UInt[64],
    dtype.Float[16], dtype.Float[32], dtype.Float[64],
]


def from_value(v, context=None, ref=None, broaden=False):
    a = to_abstract(v, context, ref)
    if broaden:
        a = _broaden(a, None)
    return a


def to_abstract(v, context=None, ref=None):
    """Translate the value to an abstract value."""
    if isinstance(v, (Primitive, Graph, MetaGraph)):
        if isinstance(v, Graph) and v.parent:
            v = GraphAndContext(v, context or Context.empty())
        if ref is not None:
            v = TrackableFunction(v, id=ref.node)
        return AbstractFunction(v)

    elif is_dataclass_type(v):
        typ = dtype.pytype_to_myiatype(v)
        typarg = AbstractScalar({
            VALUE: typ,
            TYPE: dtype.TypeType,
            SHAPE: dtype.NOSHAPE,
        })
        return AbstractFunction(
            PartialApplication(
                P.make_record,
                (typarg,)
            )
        )

    elif is_dataclass(v):
        typ = dtype.pytype_to_myiatype(type(v), v)
        new_args = {}
        for name, field in v.__dataclass_fields__.items():
            new_args[name] = to_abstract(getattr(v, name), context)
        return AbstractClass(typ.tag, new_args, typ.methods)

    elif isinstance(v, (int, float, str)):
        return AbstractScalar({
            VALUE: v,
            TYPE: dtype.pytype_to_myiatype(type(v), v),
            SHAPE: dtype.NOSHAPE,
        })

    elif isinstance(v, tuple):
        return AbstractTuple([to_abstract(elem, context) for elem in v])

    elif isinstance(v, np.ndarray):
        return AbstractArray(
            AbstractScalar({
                VALUE: ANYTHING,
                TYPE: dtype.np_dtype_to_type(str(v.dtype)),
                SHAPE: dtype.NOSHAPE
            }),
            {SHAPE: v.shape}
        )

    elif isinstance(v, list):
        if len(v) == 0:
            raise Exception('No support for empty lists yet.')
        return AbstractList(to_abstract(v[0], context))

    elif dtype.ismyiatype(v):
        return AbstractType({
            VALUE: v,
            TYPE: dtype.TypeType,
            SHAPE: dtype.NOSHAPE
        })

    else:
        typ = dtype.pytype_to_myiatype(type(v), v)
        assert dtype.ismyiatype(typ, (dtype.External, dtype.EnvType))
        return AbstractScalar({
            VALUE: v,
            TYPE: typ,
            SHAPE: dtype.NOSHAPE,
        })


class AbstractTrack(Track):
    def __init__(self,
                 engine,
                 *,
                 constructors,
                 max_depth=1):
        super().__init__(engine)
        self.constructors = {
            prim: cons()
            for prim, cons in constructors.items()
        }
        self.max_depth = max_depth

    get_inferrer_for = Overload()

    @get_inferrer_for.register
    def get_inferrer_for(self, prim: Primitive):
        return self.constructors[prim]

    @get_inferrer_for.register
    def get_inferrer_for(self, g: Graph):
        return self.get_inferrer_for(GraphAndContext(g, Context.empty()))

    @get_inferrer_for.register
    def get_inferrer_for(self, g: GraphAndContext):
        if g not in self.constructors:
            self.constructors[g] = GraphXInferrer(g.graph, g.context)
        return self.constructors[g]

    @get_inferrer_for.register
    def get_inferrer_for(self, tf: TrackableFunction):
        if tf not in self.constructors:
            self.constructors[tf] = TrackableXInferrer(
                self.get_inferrer_for(tf.fn)
            )
        return self.constructors[tf]

    @get_inferrer_for.register
    def get_inferrer_for(self, part: PartialApplication):
        return PartialXInferrer(
            self.get_inferrer_for(part.fn),
            part.args
        )

    @get_inferrer_for.register
    def get_inferrer_for(self, j: JTransformedFunction):
        return JXInferrer(
            self.get_inferrer_for(j.fn),
            j.fn
        )

    @get_inferrer_for.register
    def get_inferrer_for(self, vf: (VirtualFunction, TypedPrimitive)):
        return VirtualXInferrer(
            vf.args,
            vf.output
        )

    @get_inferrer_for.register
    def get_inferrer_for(self, df: DummyFunction):
        raise MyiaTypeError(f'Trying to call dummy')

    @get_inferrer_for.register
    def get_inferrer_for(self, mg: MetaGraph):
        if mg not in self.constructors:
            self.constructors[mg] = MetaGraphXInferrer(mg)
        return self.constructors[mg]

    async def execute(self, fn, *args):
        infs = [self.get_inferrer_for(poss)
                for poss in await fn.get()]
        argrefs = [VirtualReference(a) for a in args]
        return await execute_inferrers(self, infs, None, argrefs)

    async def infer_apply(self, ref):
        """Get the property for a ref of an Apply node."""
        ctx = ref.context
        n_fn, *n_args = ref.node.inputs
        # We await on the function node to get the inferrer
        fn_ref = self.engine.ref(n_fn, ctx)
        fn = await fn_ref.get()
        argrefs = [self.engine.ref(node, ctx) for node in n_args]

        args = [await ref.get() for ref in argrefs]

        if not isinstance(fn, AbstractFunction):
            raise Exception(f'Not a function: {fn}')

        infs = [self.get_inferrer_for(poss)
                for poss in await fn.get()]

        return await self.engine.loop.schedule(
            execute_inferrers(self, infs, ref, argrefs),
            context_map={
                infer_trace: {**infer_trace.get(), ctx: ref}
            }
        )

    async def infer_constant(self, ctref):
        """Get the property for a ref of a Constant node."""
        v = self.engine.pipeline.resources.convert(ctref.node.value)
        res = from_value(v, ctref.context, ref=ctref)
        t = res.build(TYPE)
        if dtype.ismyiatype(t, dtype.Number):
            prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            res.values[TYPE] = self.engine.loop.create_pending_from_list(
                _number_types, t, lambda: prio
            )
        return res

    def abstract_merge(self, *values):
        return reduce(self._merge, values)

    def _merge(self, x1, x2):
        return amerge(x1, x2, loop=self.engine.loop, forced=False)

    def check_predicate(self, predicate, res):
        if isinstance(predicate, tuple):
            return any(self.check_predicate(p, res) for p in predicate)
        elif dtype.ismyiatype(predicate):
            return dtype.ismyiatype(res, predicate)
        elif isinstance(predicate, type) \
                and issubclass(predicate, AbstractBase):
            return isinstance(res, predicate)
        elif callable(predicate):
            return predicate(self, res)
        else:
            raise ValueError(predicate)  # pragma: no cover

    def assert_predicate(self, predicate, res):
        if not self.check_predicate(predicate, res):
            raise MyiaTypeError(f'Expected {predicate}, not {res}')

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

    async def chkimm(self, predicate, *values):
        return await reify(self.chk(predicate, *values))


class XInferrer(Partializable):
    def __init__(self):
        self.cache = {}

    async def __call__(self, track, outref, argrefs):
        args = tuple([await ref.get() for ref in argrefs])
        if args not in self.cache:
            self.cache[args] = await self.infer(track, *args)
        return self.cache[args]

    async def infer(self, track, *args):
        raise NotImplementedError()

    def __repr__(self):
        return f'{type(self)}'


class TrackableXInferrer(XInferrer):
    def __init__(self, subinf):
        super().__init__()
        self.subinf = subinf

    async def __call__(self, track, outref, argrefs):
        args = tuple([await ref.get() for ref in argrefs])
        self.cache[args] = await self.subinf(track, outref, argrefs)
        return self.cache[args]


class BaseGraphXInferrer(XInferrer):

    def make_context(self, track, args):
        _, ctx = self._make_argkey_and_context(track, args)
        return ctx

    def _make_argkey_and_context(self, track, argvals):
        assert argvals is not None
        g = self.get_graph(track, argvals)
        argkey = tuple(argvals)
        # Update current context using the fetched properties.
        return argkey, self.context.add(g, argkey)

    async def infer(self, track, *args):
        engine = track.engine
        g = self.get_graph(track, args)
        nargs = len(g.parameters)

        if len(args) != nargs:
            raise type_error_nargs(self, nargs, len(args))

        argkey, context = self._make_argkey_and_context(track, args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(g.parameters, argkey):
            ref = engine.ref(p, context)
            engine.cache.set_value(ref, arg)

        out = engine.ref(g.return_, context)
        return await engine.get_inferred(out)


class GraphXInferrer(BaseGraphXInferrer):

    def __init__(self, graph, context):
        super().__init__()
        self._graph = graph
        if context is None:
            self.context = Context.empty()
        else:
            self.context = context.filter(graph)
        assert self.context is not None

    def get_graph(self, track, args):
        return self._graph


class MetaGraphXInferrer(BaseGraphXInferrer):

    def __init__(self, metagraph):
        super().__init__()
        self.metagraph = metagraph
        self.context = Context.empty()
        self.graph_cache = {}

    def get_graph(self, track, argvals):
        if argvals not in self.graph_cache:
            try:
                g = self.metagraph.specialize_from_abstract(argvals)
            except GraphGenerationError as err:
                raise MyiaTypeError(f'Graph gen error: {err}')
            g = track.engine.pipeline.resources.convert(g)
            self.graph_cache[argvals] = g
        return self.graph_cache[argvals]


class PartialXInferrer(XInferrer):

    def __init__(self, fn, args):
        super().__init__()
        self.fn = fn
        self.args = args

    async def __call__(self, track, outref, argrefs):
        argvals = tuple([await ref.get() for ref in argrefs])
        if argvals not in self.cache:
            args = tuple(VirtualReference(arg)
                         for arg in self.args + argvals)
            self.cache[argvals] = await self.fn(track, outref, args)
        return self.cache[argvals]


class VirtualXInferrer(XInferrer):

    def __init__(self, args, output):
        super().__init__()
        self.args = args
        self.output = output

    async def infer(self, track, *args):
        if len(args) != len(self.args):
            raise MyiaTypeError('Wrong number of arguments')
        for given, expected in zip(args, self.args):
            track.abstract_merge(given, expected)
        return self.output


def _jinv(x):
    if isinstance(x, AbstractJTagged):
        return x.element
    else:
        raise MyiaTypeError('Expected JTagged')


def _jtag(x):
    if isinstance(x, AbstractFunction):
        v = x.values[VALUE]
        if isinstance(v, Possibilities):
            return AbstractFunction(*[JTransformedFunction(poss)
                                      for poss in v])
    return AbstractJTagged(x)


class JXInferrer(XInferrer):

    def __init__(self, fn, orig_fn):
        super().__init__()
        self.fn = fn
        self.orig_fn = orig_fn

    async def __call__(self, track, outref, argrefs):
        args = tuple([await ref.get() for ref in argrefs])
        if args not in self.cache:
            jinv_args = tuple(_jinv(a) for a in args)
            jinv_argrefs = tuple(VirtualReference(arg)
                                 for arg in jinv_args)
            res = await self.fn(track, None, jinv_argrefs)
            res_wrapped = _jtag(res)
            orig_fn = AbstractFunction(self.orig_fn)
            # bparams = [sensitivity_transform(self.orig_fn)]
            bparams = [sensitivity_transform(orig_fn)]
            bparams += [sensitivity_transform(a) for a in args]
            bparams_final = AbstractTuple(bparams)
            bprop = AbstractFunction(
                VirtualFunction(
                    (sensitivity_transform(res),),
                    bparams_final
                )
            )
            self.cache[args] = AbstractTuple([res_wrapped, bprop])
        return self.cache[args]


async def _xinf_helper(track, inf, outref, argrefs, p):
    result = await inf(track, outref, argrefs)
    p.resolve_to(result)


async def execute_inferrers(track, inferrers, outref, argrefs):
    if len(inferrers) == 1:
        inf, = inferrers
        return await inf(track, outref, argrefs)

    else:
        pending = []
        for inf in inferrers:
            p = track.engine.loop.create_pending(
                resolve=None,
                priority=lambda: None
            )
            pending.append(p)
            track.engine.loop.schedule(
                _xinf_helper(track, inf, outref, argrefs, p)
            )

        return bind(track.engine.loop, None, [], pending)
