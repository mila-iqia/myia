
from functools import reduce

from .. import dtype, dshape
from ..infer import Track, MyiaTypeError, Context
from ..infer.core import Pending
from ..infer.graph_infer import type_error_nargs, VirtualReference
from ..infer.utils import infer_trace
from ..ir import Graph, MetaGraph, GraphGenerationError
from ..prim import Primitive, ops as P
from ..prim.py_implementations import typeof
from ..utils import as_frozen, Var, RestrictedVar, Overload, Partializable, \
    is_dataclass_type

from .base import from_vref, shapeof, AbstractScalar, Possibilities, \
    ABSENT, GraphAndContext, AbstractBase, amerge, bind, PartialApplication, \
    reify, JTransformedFunction, AbstractJTagged, AbstractTuple, \
    sensitivity_transform, VirtualFunction, AbstractFunction, \
    VALUE, TYPE, SHAPE, REF


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
                 constructors,
                 max_depth=1):
        super().__init__(engine, name)
        self.constructors = {
            prim: cons()
            for prim, cons in constructors.items()
        }
        self.max_depth = max_depth
        self.subtracks = [VALUE, TYPE, SHAPE]

    get_inferrer_for = Overload()

    @get_inferrer_for.register
    def get_inferrer_for(self, prim: Primitive, args):
        return self.constructors[prim]

    @get_inferrer_for.register
    def get_inferrer_for(self, g: Graph, args):
        return self.get_inferrer_for(GraphAndContext(g, Context.empty()), args)

    @get_inferrer_for.register
    def get_inferrer_for(self, g: GraphAndContext, args):
        if g not in self.constructors:
            self.constructors[g] = GraphXInferrer(g.graph, g.context)
        return self.constructors[g]

    @get_inferrer_for.register
    def get_inferrer_for(self, part: PartialApplication, args):
        if part not in self.constructors:
            self.constructors[part] = PartialXInferrer(
                self.get_inferrer_for(part.fn, (*part.args, *args)),
                part.args
            )
        return self.constructors[part]

    @get_inferrer_for.register
    def get_inferrer_for(self, j: JTransformedFunction, args):
        if j not in self.constructors:
            self.constructors[j] = JXInferrer(
                self.get_inferrer_for(j.fn, args),
                j.fn
            )
        return self.constructors[j]

    @get_inferrer_for.register
    def get_inferrer_for(self, vf: VirtualFunction, args):
        if vf not in self.constructors:
            self.constructors[vf] = VirtualXInferrer(
                vf.args,
                vf.output
            )
        return self.constructors[vf]

    @get_inferrer_for.register
    def get_inferrer_for(self, mg: MetaGraph, args):
        try:
            g = mg.specialize_from_abstract(args)
        except GraphGenerationError as err:
            raise MyiaTypeError(f'Graph gen error: {err}')
        g = self.engine.pipeline.resources.convert(g)
        return self.get_inferrer_for(g, args)

    async def execute(self, fn, *args):
        infs = [self.get_inferrer_for(poss, args)
                for poss in fn.values[VALUE]]
        argrefs = [VirtualReference({'abstract': a}) for a in args]
        return await execute_inferrers(self, infs, None, argrefs)

    async def infer_apply(self, ref):
        """Get the property for a ref of an Apply node."""
        ctx = ref.context
        n_fn, *n_args = ref.node.inputs
        # We await on the function node to get the inferrer
        fn_ref = self.engine.ref(n_fn, ctx)
        fn = await fn_ref[self.name]
        argrefs = [self.engine.ref(node, ctx) for node in n_args]

        args = [await ref['abstract'] for ref in argrefs]

        if not isinstance(fn, AbstractFunction):
            raise Exception(f'Not a function: {fn}')

        infs = [self.get_inferrer_for(poss, args)
                for poss in fn.values[VALUE]]

        return await self.engine.loop.schedule(
            execute_inferrers(self, infs, ref, argrefs),
            context_map={
                infer_trace: {**infer_trace.get(), ctx: ref}
            }
        )

    async def infer_constant(self, ctref):
        """Get the property for a ref of a Constant node."""
        v = self.engine.pipeline.resources.convert(ctref.node.value)
        res = self.from_value(v, ctref.context)
        t = res.build(TYPE)
        if dtype.ismyiatype(t, dtype.Number):
            prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            res.values[TYPE] = self.engine.loop.create_pending_from_list(
                _number_types, t, prio
            )
            # v = RestrictedVar(_number_types)
            # prio = 1 if dtype.ismyiatype(t, dtype.Float) else 0
            # res.values[TYPE] = self.engine.loop.create_var(v, t, prio)
        res.values[REF].setdefault(ctref.context, ctref.node)
        return res

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return AbstractFunction(v)
        elif isinstance(v, Graph):
            if v.parent:
                v = GraphAndContext(v, context)
            return AbstractFunction(v)
        elif isinstance(v, MetaGraph):
            return AbstractFunction(v)
        elif is_dataclass_type(v):
            # rec = self.constructors[P.make_record]()
            # typ = dtype.pytype_to_myiatype(v)
            # vref = self.engine.vref({VALUE: typ, TYPE: TypeType})
            # return PartialInferrer(self, rec, [vref])

            typ = dtype.pytype_to_myiatype(v)
            typarg = AbstractScalar({
                VALUE: typ,
                TYPE: dtype.TypeType,
                SHAPE: dshape.NOSHAPE,
            })
            return AbstractFunction(
                PartialApplication(
                    P.make_record,
                    (typarg,)
                )
            )
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
                # if dtype.ismyiatype(predicate):
                #     value.equiv.add(predicate)
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

    def build(self, field):
        return None

    async def __call__(self, track, outref, argrefs):
        args = tuple([await ref['abstract'] for ref in argrefs])
        if args not in self.cache:
            self.cache[args] = await self.infer(track, *args)
        return self.cache[args]

    async def infer(self, track, *args):
        raise NotImplementedError()

    def __repr__(self):
        return f'{type(self)}'


class GraphXInferrer(XInferrer):

    def __init__(self, graph, context):
        super().__init__()
        self._graph = graph
        if context is None:
            self.context = Context.empty()
        else:
            self.context = context.filter(graph)
        assert self.context is not None

    def make_context(self, track, args):
        _, ctx = self._make_argkey_and_context(track, args)
        return ctx

    def _make_argkey_and_context(self, track, argvals):
        assert argvals is not None
        argkey = as_frozen(argvals)
        # Update current context using the fetched properties.
        return argkey, self.context.add(self._graph, argkey)

    async def infer(self, track, *args):
        engine = track.engine
        g = self._graph
        nargs = len(g.parameters)

        if len(args) != nargs:
            raise type_error_nargs(self, nargs, len(args))

        argkey, context = self._make_argkey_and_context(track, args)

        # We associate each parameter of the Graph with its value for each
        # property, in the context we built.
        for p, arg in zip(g.parameters, argkey):
            arg.values[REF].setdefault(context, p)
            ref = engine.ref(p, context)
            engine.cache.set_value(('abstract', ref), arg)

        out = engine.ref(g.return_, context)
        return await engine.get_inferred('abstract', out)


class PartialXInferrer(XInferrer):

    def __init__(self, fn, args):
        super().__init__()
        self.fn = fn
        self.args = args

    async def __call__(self, track, outref, argrefs):
        argvals = tuple([await ref['abstract'] for ref in argrefs])
        if argvals not in self.cache:
            args = tuple(VirtualReference({'abstract': arg})
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
    if isinstance(x, AbstractFunction):
        v = x.values[VALUE]
        if isinstance(v, Possibilities):
            pass
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
        args = tuple([await ref['abstract'] for ref in argrefs])
        if args not in self.cache:
            jinv_args = tuple(_jinv(a) for a in args)
            jinv_argrefs = tuple(VirtualReference({'abstract': arg})
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
    result.values[REF].setdefault(outref.context, outref.node)
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
                priority=None
            )
            pending.append(p)
            track.engine.loop.schedule(
                _xinf_helper(track, inf, outref, argrefs, p)
            )

        return bind(track.engine.loop, None, [], pending)
