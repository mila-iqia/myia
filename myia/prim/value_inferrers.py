"""Definitions of value inference for primitives."""


import asyncio

from functools import partial

from ..dtype import pytype_to_myiatype, TypeType
from ..infer import ValueWrapper, InferenceError, PartialInferrer, \
    ANYTHING, Inferrer, GraphInferrer, register_inferrer, Track, \
    unwrap, MetaGraphInferrer
from ..ir import Graph, MetaGraph
from ..utils import is_dataclass

from . import ops as P
from .inferrer_utils import static_getter
from .ops import Primitive
from .py_implementations import hastype_helper


class LimitedValue(ValueWrapper):
    """Value associated to a count.

    Attributes:
        value: The value.
        count: A count, which is intended to decrease to zero as we execute
            functions and go down the stack, at which point we broaden the
            value to ANYTHING.

    """

    @classmethod
    def min_count(cls, lvs):
        """Return the minimum count of all LimitedValues."""
        return min(lv.count for lv in lvs)

    def __init__(self, value, count):
        """Initialize a LimitedValue."""
        super().__init__(value)
        self.count = count


def limited(v, count):
    """Wrap the value if count > 0, else return ANYTHING."""
    if count <= 0:
        return ANYTHING
    else:
        return LimitedValue(v, count)


class PrimitiveValueInferrer(Inferrer):
    """Infer the return value of a function using its implementation.

    If any input is ANYTHING, the return value will also be ANYTHING.
    The implementation will not be called.
    """

    def __init__(self, track, prim, impl):
        """Initialize a PrimitiveValueInferrer."""
        super().__init__(track, prim)
        self.impl = impl

    async def infer(self, *refs):
        """Infer the return value of a function using its implementation."""
        coros = [ref.get_raw('value') for ref in refs]
        args = await asyncio.gather(*coros, loop=self.engine.loop)
        if any(arg is ANYTHING for arg in args):
            return ANYTHING
        else:
            args_unwrapped = [unwrap(arg) for arg in args]
            try:
                v = self.impl(*args_unwrapped)
                try:
                    # Throw away non-hashable results
                    # Use a whitelist of types instead?
                    hash(v)
                except TypeError:
                    return ANYTHING
            except Exception as e:
                raise InferenceError(e, refs=refs)
            n = LimitedValue.min_count(args)
            return limited(v, n)


value_inferrer_constructors = {}


class ValueTrack(Track):
    """Infer the value of a constant.

    Note: the value of a Primitive or of a Graph is an Inferrer.

    Attributes:
        max_depth: How deep in the call stack to infer values. Past that
            depth, inferred values are degraded to ANYTHING.
        implementations: A map of primitives to implementations.
        constructors: A map of Inferrer constructors. Each constructor
            takes an engine as argument and returns an Inferrer. These
            will be used to infer values for primitives.

    """

    def __init__(self,
                 engine,
                 name,
                 max_depth=10,
                 constructors=value_inferrer_constructors):
        """Initialize a ValueTrack."""
        super().__init__(engine, name)
        self.implementations = engine.pipeline.resources.py_implementations
        self.constructors = constructors
        self.max_depth = max_depth

    def wrap(self, v):
        """Produce a LimitedValue for v, with a maximal count."""
        return LimitedValue(v, self.max_depth)

    def default(self, values):
        """Default value for ValueTrack."""
        return ANYTHING

    def from_value(self, v, context):
        """Infer the value of a constant."""
        if isinstance(v, Primitive):
            if v in self.constructors:
                inf = self.constructors[v](self)
            else:
                inf = PrimitiveValueInferrer(
                    self, v, self.implementations[v]
                )
        elif isinstance(v, Graph):
            inf = GraphInferrer(self, v, context)
        elif isinstance(v, MetaGraph):
            inf = MetaGraphInferrer(self, v)
        elif is_dataclass(v):
            p = P.make_record
            recinf = PrimitiveValueInferrer(self, p, self.implementations[p])
            typ = pytype_to_myiatype(v)
            vref = self.engine.vref({'value': limited(typ, self.max_depth),
                                     'type': TypeType()})
            return PartialInferrer(self, recinf, [vref])
        elif v is ANYTHING:
            return v
        else:
            return self.wrap(v)

        return self.wrap(inf)

    def broaden(self, v):
        """Broaden the value if we reach a certain depth in the stack."""
        if v is ANYTHING:
            return v
        else:
            return limited(v.value, v.count - 1)


########################
# Default constructors #
########################


value_inferrer = partial(register_inferrer,
                         constructors=value_inferrer_constructors)


@value_inferrer(P.if_, nargs=3)
async def infer_value_if(engine, cond, tb, fb):
    """Infer the return value of if.

    If the condition is ANYTHING, the return value is also ANYTHING,
    regardless of whether the value of either or both branches can
    be inferred.
    """
    v = await cond['value']
    if v is True:
        fn = await tb['value']
    elif v is False:
        fn = await fb['value']
    elif v is ANYTHING:
        # Note: we do not infer the values for the branches at all.
        # If we did, we may encounter recursion and deadlock.
        return ANYTHING

    return await fn()


@value_inferrer(P.switch, nargs=3)
async def infer_value_switch(track, cond, tb, fb):
    """Infer the return type of switch."""
    v = await cond['value']
    if v is True:
        return await tb.get_raw('value')
    elif v is False:
        return await fb.get_raw('value')
    elif v is ANYTHING:
        # Note: we do not infer the values for the branches at all.
        # If we did, we may encounter recursion and deadlock.
        return ANYTHING


@value_inferrer(P.partial, nargs=None)
async def infer_value_partial(engine, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['value']
    return PartialInferrer(engine, fn_t, args)


@value_inferrer(P.hastype, nargs=2)
async def infer_value_hastype(track, x, t):
    """Infer the return value of hastype."""
    x_t = await x['type']
    t_v = await t['value']
    if t_v is ANYTHING:
        raise InferenceError('Second argument to hastype must be constant.')
    # TODO: Find a good way to carry ValueTrack.max_depth to here
    # Instead of defaulting to 1
    return limited(hastype_helper(x_t, t_v), track.max_depth)


@value_inferrer(P.typeof, nargs=1)
async def infer_value_typeof(track, x):
    """Infer the return value of typeof."""
    # TODO: Find a good way to carry ValueTrack.max_depth to here
    # Instead of defaulting to 1
    return limited(await x['type'], track.max_depth)


@value_inferrer(P.shape, nargs=1)
async def infer_value_shape(track, ary):
    """Infer the return value of shape."""
    shp = await ary['shape']
    if any(s is ANYTHING for s in shp):
        return ANYTHING
    # TODO: Should propagate ValueTrack.max_depth here
    return limited(shp, 1)


@value_inferrer(P.resolve, nargs=2)
async def infer_value_resolve(track, data, item):
    """Infer the return value of resolve."""
    return await static_getter(track, data, item, lambda x, y: x[y])


@value_inferrer(P.getattr, nargs=2)
async def infer_value_getattr(track, data, item):
    """Infer the return value of getattr."""
    return await static_getter(track, data, item, getattr)
