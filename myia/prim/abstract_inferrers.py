
import numpy
from asyncio import Future
from functools import partial
from . import ops as P
from .. import dtype, dshape
from ..infer import Track, register_inferrer, ANYTHING, InferenceError, \
    MyiaTypeError
from ..infer.utils import infer_trace
from ..ir import Graph
from ..prim import Primitive, ops as P
from ..utils import overload, UNKNOWN, Named, as_frozen
from ..abstract import \
    AbstractBase, \
    AbstractValue, \
    AbstractTuple, \
    AbstractArray, \
    AbstractList, \
    AbstractClass, \
    ABSENT, \
    AbstractTrack, \
    XInferrer, \
    GraphXInferrer, \
    abstract_inferrer_constructors


abstract_inferrer = partial(register_inferrer,
                            constructors=abstract_inferrer_constructors)


# class AbstractTrack(Track):
#     def __init__(self,
#                  engine,
#                  name,
#                  *,
#                  constructors=abstract_inferrer_constructors):
#         super().__init__(engine, name)
#         self.constructors = constructors
#         self.subtracks = ['value', 'type', 'shape']

#     async def infer_apply(self, ref):
#         """Get the property for a ref of an Apply node."""
#         ctx = ref.context
#         n_fn, *n_args = ref.node.inputs
#         # We await on the function node to get the inferrer
#         fn_ref = self.engine.ref(n_fn, ctx)
#         inf = await fn_ref[self.name]
#         argrefs = [self.engine.ref(node, ctx) for node in n_args]
#         if not isinstance(inf, XInferrer):
#             raise MyiaTypeError(
#                 f'Trying to call a non-callable type: {inf}',
#                 refs=[fn_ref],
#                 app=ref
#             )
#         return await self.engine.loop.schedule(
#             inf(self, *argrefs),
#             context_map={
#                 infer_trace: {**infer_trace.get(), ctx: ref}
#             }
#         )

#     async def infer_constant(self, ctref):
#         v = ctref.node.value
#         if isinstance(v, Graph):
#             return GraphXInferrer(v, ctref.context)
#         elif isinstance(v, Primitive):
#             return self.constructors[v]()
#         else:
#             return from_vref(
#                 v,
#                 dtype.pytype_to_myiatype(type(v), v),
#                 dshape.NOSHAPE,
#             )

#     def from_value(self, v, context):
#         return 8911

#     def from_external(self, t):
#         return 21

#     def default(self, values):
#         # return AbstractValue(values)
#         # return AbstractTuple((1, 2, 3))
#         return from_vref(
#             values['value'],
#             values['type'],
#             values['shape'],
#         )


# class XInferrer:
#     def __init__(self):
#         self.cache = {}

#     async def __call__(self, track, *refs):
#         args = tuple([await ref['abstract'] for ref in refs])
#         if args not in self.cache:
#             self.cache[args] = await self.infer(track, *args)
#         return self.cache[args]

#     async def infer(self, track, *args):
#         raise NotImplementedError()


# class GraphXInferrer(XInferrer):

#     def __init__(self, graph, context, broaden=True):
#         super().__init__()
#         self._graph = graph
#         self.broaden = broaden
#         if context is None:
#             self.context = Context.empty()
#         else:
#             self.context = context.filter(graph)
#         assert self.context is not None

#     async def make_graph(self, args):
#         return self._graph

#     async def make_context(self, track, args):
#         _, ctx = await self._make_argkey_and_context(track, args)
#         return ctx

#     async def _make_argkey_and_context(self, track, args):
#         engine = track.engine
#         g = await self.make_graph(args)
#         argvals = []
#         for arg in args:
#             argval = {}
#             for track_name, track in engine.tracks.items():
#                 result = await engine.get_inferred(track_name, arg)
#                 if self.broaden and not g.flags.get('flatten_inference'):
#                     result = track.broaden(result)
#                 argval[track_name] = result
#             argvals.append(argval)

#         # Update current context using the fetched properties.
#         argkey = as_frozen(argvals)
#         return argkey, self.context.add(g, argkey)

#     async def __call__(self, track, *args):
#         if args not in self.cache:
#             self.cache[args] = await self.infer(track, *args)
#         return self.cache[args]

#     async def infer(self, track, *args):
#         engine = track.engine
#         g = await self.make_graph(args)
#         nargs = len(g.parameters)

#         if len(args) != nargs:
#             raise type_error_nargs(self.identifier, nargs, len(args))

#         argkey, context = await self._make_argkey_and_context(track, args)

#         # We associate each parameter of the Graph with its value for each
#         # property, in the context we built.
#         for p, arg in zip(g.parameters, argkey):
#             for track, v in arg:
#                 ref = engine.ref(p, context)
#                 engine.cache.set_value((track, ref), v)

#         out = engine.ref(g.return_, context)
#         return await engine.get_inferred('abstract', out)


# class IdentityXInferrer(XInferrer):
#     async def infer(self, track, x):
#         return x


# class PrimitiveXInferrer(XInferrer):
#     async def infer(self, track, *args):
#         rval = await self.infer_structure(track, *args)
#         for t in track.subtracks:
#             method = getattr(self, f'infer_{t}')
#             rval.values[t] = await method(track, *args)
#         return rval

#     async def infer_structure(self, track, *args):
#         raise NotImplementedError()

#     async def infer_value(self, track, *args):
#         raise NotImplementedError()

#     async def infer_type(self, track, *args):
#         raise NotImplementedError()

#     async def infer_shape(self, track, *args):
#         raise NotImplementedError()


# class StructuralXInferrer(PrimitiveXInferrer):

#     async def infer_value(self, track, *args):
#         return ANYTHING

#     async def infer_type(self, track, *args):
#         return ABSENT

#     async def infer_shape(self, track, *args):
#         return ABSENT


# class MakeTupleXInferrer(StructuralXInferrer):
#     async def infer_structure(self, track, *args):
#         return AbstractTuple(args)


# class TupleGetitemXInferrer(XInferrer):
#     async def infer(self, track, arg, idx):
#         i = idx.values['value']
#         return arg.elements[i]


# class ScalarXInferrer(PrimitiveXInferrer):

#     async def infer_structure(self, track, *args):
#         assert all(isinstance(arg, AbstractValue) for arg in args)
#         return AbstractValue({})

#     async def infer_value(self, track, *args):
#         values = [arg.values['value'] for arg in args]
#         if any(v is ANYTHING for v in values):
#             return ANYTHING
#         return self.impl(*values)

#     async def infer_type(self, track, *args):
#         ts = [arg.values['type'] for arg in args]
#         assert all(dtype.ismyiatype(t, dtype.Number) for t in ts)
#         assert all(t == ts[0] for t in ts)
#         return ts[0]

#     async def infer_shape(self, track, *args):
#         return dshape.NOSHAPE


# class AddXInferrer(ScalarXInferrer):
#     def impl(self, x, y):
#         return x + y


# class MulXInferrer(ScalarXInferrer):
#     def impl(self, x, y):
#         return x * y


# abstract_inferrer_constructors[P.scalar_add] = AddXInferrer
# abstract_inferrer_constructors[P.scalar_mul] = MulXInferrer
# abstract_inferrer_constructors[P.identity] = IdentityXInferrer
# abstract_inferrer_constructors[P.return_] = IdentityXInferrer
# abstract_inferrer_constructors[P.make_tuple] = MakeTupleXInferrer
# abstract_inferrer_constructors[P.tuple_getitem] = TupleGetitemXInferrer



# @abstract_inferrer(P.switch, nargs=3)
# async def infer_abstract_switch(track, cond, tb, fb):
#     pass
#     # """Infer the return type of if."""
#     # await track.check(Bool, cond)
#     # tb_inf = await tb['type']
#     # fb_inf = await fb['type']
#     # v = await cond['value']
#     # if v is True:
#     #     # We only visit the first branch if the condition is provably true
#     #     return await tb_inf()
#     # elif v is False:
#     #     # We only visit the second branch if the condition is provably false
#     #     return await fb_inf()
#     # elif v is ANYTHING:
#     #     # The first branch to finish will return immediately. When the other
#     #     # branch finishes, its result will be checked against the other.
#     #     return await track.assert_same(tb_inf(), fb_inf(), refs=[tb, fb])
#     # else:
#     #     raise AssertionError("Invalid condition value for if")



# @overload(bootstrap=True)
# def from_value(self, xs: tuple):
#     return AbstractTuple([self(x) for x in xs])


# @overload
# def from_value(self, xs: list):
#     elems = [self(x) for x in xs]
#     return AbstractList(xxx)


# @overload
# def from_value(self, xs: numpy.ndarray):
#     elems = [self(x) for x in xs]
#     return AbstractArray(xxx)



# @overload(bootstrap=True)
# def from_type(self, t: dtype.Tuple):
#     return AbstractTuple([self(tt) for tt in t.elements])


# @overload
# def from_type(self, t: dtype.Array):
#     return AbstractArray(self(t.elements))


# @overload
# def from_type(self, t: dtype.List):
#     return AbstractList(self(t.elements))


# @overload(bootstrap=True)
# def from_vref(self, v, t: dtype.Tuple, s):
#     elems = []
#     for i, tt in enumerate(t.elements):
#         vv = v[i] if isinstance(v, tuple) else ANYTHING
#         ss = s.shape[i]
#         elems.append(self(vv, tt, ss))
#     return AbstractTuple(elems)


# @overload
# def from_vref(self, v, t: dtype.Array, s):
#     vv = ANYTHING
#     tt = t.elements
#     ss = dshape.NOSHAPE
#     return AbstractArray(self(vv, tt, ss), {'shape': s})


# @overload
# def from_vref(self, v, t: dtype.List, s):
#     vv = ANYTHING
#     tt = t.element_type
#     ss = s.shape
#     return AbstractList(self(vv, tt, ss), {})


# @overload
# def from_vref(self, v, t: (dtype.Number, dtype.Bool, dtype.External), s):
#     return AbstractValue({'value': v, 'type': t, 'shape': s})


# @overload
# def from_vref(self, v, t: dtype.Class, s):
#     attrs = {}
#     for k, tt in t.attributes.items():
#         vv = ANYTHING if v in (ANYTHING, UNKNOWN) else getattr(v, k)
#         ss = ANYTHING if s in (ANYTHING, UNKNOWN) else s.shape[k]
#         attrs[k] = self(vv, tt, ss)
#     return AbstractClass(
#         t.tag,
#         attrs,
#         t.methods
#         # {'value': v, 'type': t, 'shape': s}
#     )


# @overload
# def from_vref(self, v, t: dtype.TypeMeta, s):
#     return self[t](v, t, s)
