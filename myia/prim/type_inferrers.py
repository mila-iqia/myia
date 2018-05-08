"""Definitions of type inference for primitives."""


from functools import partial

from ..dtype import Int, Bool, Float, Tuple, List
from ..infer import ANYTHING, Inferrer, GraphInferrer, \
    MyiaTypeError, register_inferrer, Track
from ..ir import Graph

from . import ops as P
from .ops import Primitive


def typeof(v):
    """Return the type of v."""
    if isinstance(v, bool):
        return Bool()
    elif isinstance(v, int):
        return Int(64)
    elif isinstance(v, float):
        return Float(64)
    elif isinstance(v, tuple):
        return Tuple(map(typeof, v))
    else:
        raise TypeError(f'Untypable value: {v}')  # pragma: no cover


type_inferrer_constructors = {}


class TypeTrack(Track):
    """Infer the type of a constant.

    Note: the type of a Primitive or of a Graph is an Inferrer.

    Attributes:
        constructors: A map of Inferrer constructors. Each constructor
            takes an engine as argument and returns an Inferrer. These
            will be used to infer types for primitives.

    """

    def __init__(self,
                 engine,
                 name,
                 *,
                 constructors=type_inferrer_constructors):
        """Initialize a TypeTrack.

        If a TypeTrack is present, it is always required.
        """
        super().__init__(engine, name)
        self.constructors = constructors

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v](self.engine)
        elif isinstance(v, Graph):
            return GraphInferrer(self.engine, 'type', v, context)
        else:
            return typeof(v)

    def default(self):
        """Return a default type; this method raises an exception."""
        raise Exception('There is no default value for the type track.') \
            # pragma: no cover


########################
# Default constructors #
########################


type_inferrer = partial(register_inferrer,
                        constructors=type_inferrer_constructors)


@type_inferrer(P.if_, nargs=3)
async def infer_type_if(engine, cond, tb, fb):
    """Infer the return type of if."""
    cond_t = await cond['type']
    if cond_t != Bool():
        raise MyiaTypeError('Condition for if must be a boolean')
    tb_inf = await tb['type']
    fb_inf = await fb['type']
    if not isinstance(tb_inf, Inferrer) or not isinstance(fb_inf, Inferrer):
        raise MyiaTypeError('Both branches of if primitive must be thunks') \
            # pragma: no cover
    v = await cond['value']
    if v is True:
        # We only visit the first branch if the condition is provably true
        return await tb_inf()
    elif v is False:
        # We only visit the second branch if the condition is provably false
        return await fb_inf()
    elif v is ANYTHING:
        # The first branch to finish will return immediately. When the other
        # branch finishes, its result will be checked against the other.
        return await engine.assert_same('type', tb_inf(), fb_inf())


@type_inferrer(P.cons_tuple, nargs=2)
async def infer_type_cons_tuple(engine, x, y):
    """Infer the return type of cons_tuple."""
    x_t = await x['type']
    y_t = await y['type']
    if not isinstance(y_t, Tuple):
        raise MyiaTypeError('cons_tuple on non-tuple')  # pragma: no cover
    return Tuple([x_t, *y_t.elements])


@type_inferrer(P.head, nargs=1)
async def infer_type_head(engine, tup):
    """Infer the return type of head."""
    tup_t = await tup['type']
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('head of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('head on empty tuple')
    return tup_t.elements[0]


@type_inferrer(P.tail, nargs=1)
async def infer_type_tail(engine, tup):
    """Infer the return type of tail."""
    tup_t = await tup['type']
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('tail of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('tail on empty tuple')
    return Tuple(tup_t.elements[1:])


@type_inferrer(P.getitem, nargs=2)
async def infer_type_getitem(engine, seq, idx):
    """Infer the return type of getitem."""
    seq_t = await seq['type']
    idx_t = await idx['type']
    if not isinstance(idx_t, Int):
        raise MyiaTypeError('Expected Int for index')

    if isinstance(seq_t, Tuple):
        idx_v = await idx['value']
        if idx_v is ANYTHING:
            raise MyiaTypeError('Tuples must be indexed with a constant')
        return seq_t.elements[idx_v]
    elif isinstance(seq_t, List):
        return seq_t.element_type
    else:
        raise MyiaTypeError('Wrong seq type for getitem')


@type_inferrer(P.not_, nargs=1)
async def infer_type_not(engine, x):
    """Infer the return type of not."""
    x_t = await x['type']
    if x_t != Bool():
        raise MyiaTypeError('Expected Bool for not.')
    return Bool()


@type_inferrer(P.eq, P.ne, nargs=2)
async def infer_type_generic_compare(engine, x, y):
    """Infer the return type of a generic comparison operator."""
    await engine.assert_same('type', x, y)
    return Bool()


@type_inferrer(P.lt, P.gt, P.le, P.ge, nargs=2)
async def infer_type_arith_compare(engine, x, y):
    """Infer the return type of an arithmetic comparison operator."""
    t = await engine.assert_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return Bool()


@type_inferrer(P.uadd, P.usub, nargs=1)
async def infer_type_arith_unary(engine, x):
    """Infer the return type of a unary arithmetic operator."""
    t = await x['type']
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


@type_inferrer(P.add, P.sub, P.mul, P.div, P.mod, P.pow, nargs=2)
async def infer_type_arith_bin(engine, x, y):
    """Infer the return type of a binary arithmetic operator."""
    t = await engine.assert_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t
