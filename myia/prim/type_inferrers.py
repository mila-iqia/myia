
from ..dtype import Int, Bool, Float, Tuple, List
from ..infer import ANYTHING, Inferrer, PrimitiveInferrer, GraphInferrer, \
    MyiaTypeError
from ..ir import Graph

from . import ops as P
from .ops import Primitive


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
        raise TypeError(f'Untypable value: {v}')  # pragma: no cover


class TypeTrack:

    def __init__(self, constructors):
        self.constructors = constructors

    async def __call__(self, engine, ct):
        v = ct.node.value
        if isinstance(v, Primitive):
            return self.constructors[v](engine)
        elif isinstance(v, Graph):
            return GraphInferrer(engine, 'type', v, ct.context)
        else:
            return typeof(ct.node.value)


type_inferrer_constructors = {}
infer_type_constant = TypeTrack(type_inferrer_constructors)


def type_inferrer(prim, nargs):
    def deco(fn):
        def constructor(engine):
            return PrimitiveInferrer(engine, prim, nargs, fn)
        type_inferrer_constructors[prim] = constructor
        return fn
    return deco


@type_inferrer(P.if_, 3)
async def infer_type_if(engine, cond, tb, fb):
    cond_t = await engine.get('type', cond)
    if cond_t != Bool():
        raise MyiaTypeError('Condition for if must be a boolean')
    v = await engine.get('value', cond)
    tb_inf = await engine.get('type', tb)
    fb_inf = await engine.get('type', fb)
    if not isinstance(tb_inf, Inferrer) or not isinstance(fb_inf, Inferrer):
        raise MyiaTypeError('Both branches of if primitive must be thunks') \
            # pragma: no cover
    if v is True:
        return await tb_inf()
    elif v is False:
        return await fb_inf()
    elif v is ANYTHING:
        return await engine.assert_same('type', tb_inf(), fb_inf())


@type_inferrer(P.cons_tuple, 2)
async def infer_type_cons_tuple(engine, x, y):
    x_t = await engine.get('type', x)
    y_t = await engine.get('type', y)
    if not isinstance(y_t, Tuple):
        raise MyiaTypeError('cons_tuple on non-tuple')  # pragma: no cover
    return Tuple([x_t, *y_t.elements])


@type_inferrer(P.head, 1)
async def infer_type_head(engine, tup):
    tup_t = await engine.get('type', tup)
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('head of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('head on empty tuple')
    return tup_t.elements[0]


@type_inferrer(P.tail, 1)
async def infer_type_tail(engine, tup):
    tup_t = await engine.get('type', tup)
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('tail of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('tail on empty tuple')
    return Tuple(tup_t.elements[1:])


@type_inferrer(P.getitem, 2)
async def infer_type_getitem(engine, seq, idx):
    seq_t = await engine.get('type', seq)
    idx_t = await engine.get('type', idx)
    if not isinstance(idx_t, Int):
        raise MyiaTypeError('Expected Int for index')

    if isinstance(seq_t, Tuple):
        idx_v = await engine.get('value', idx)
        if idx_v is ANYTHING:
            raise MyiaTypeError('Tuples must be indexed with a constant')
        return seq_t.elements[idx_v]
    elif isinstance(seq_t, List):
        return seq_t.element_type
    else:
        raise MyiaTypeError('Wrong seq type for getitem')


async def infer_type_compare(engine, x, y):
    t = await engine.assert_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return Bool()


async def infer_type_arith_unary(engine, x):
    t = await engine.get('type', x)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


async def infer_type_arith_bin(engine, x, y):
    t = await engine.assert_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


def _register_inferrer(prim, nargs, fn):
    def construct(engine):
        return PrimitiveInferrer(engine, prim, nargs, fn)
    type_inferrer_constructors[prim] = construct


for op in [P.add, P.sub, P.mul, P.div, P.mod, P.pow]:
    _register_inferrer(op, 2, infer_type_arith_bin)


for op in [P.uadd, P.usub]:
    _register_inferrer(op, 1, infer_type_arith_unary)


for op in [P.eq, P.lt, P.gt, P.ne, P.le, P.ge]:
    _register_inferrer(op, 2, infer_type_compare)
