
from ..dtype import Int, Bool, Float, Tuple, List
from ..infer import ANYTHING, PrimitiveInferrer, GraphInferrer, \
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
        raise TypeError(f'Untypable value: {v}')


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


def type_inferrer(prim):
    def deco(fn):
        def constructor(engine):
            return PrimitiveInferrer(engine, 'type', fn)
        type_inferrer_constructors[prim] = constructor
        return fn
    return deco


@type_inferrer(P.if_)
async def infer_type_if(engine, cond, tb, fb):
    assert await engine.get('type', cond) == Bool()
    v = await engine.get('value', cond)
    tb_res = (await engine.get('type', tb))()
    fb_res = (await engine.get('type', fb))()
    if v is True:
        return await tb_res
    elif v is False:
        return await fb_res
    elif v is ANYTHING:
        return await engine.force_same('type', tb_res, fb_res)


@type_inferrer(P.cons_tuple)
async def infer_type_cons_tuple(engine, x, y):
    x_t = await engine.get('type', x)
    y_t = await engine.get('type', y)
    assert isinstance(y_t, Tuple)
    return Tuple([x_t, *y_t.elements])


@type_inferrer(P.getitem)
async def infer_type_getitem(engine, seq, idx):
    seq_t = await engine.get('type', seq)
    idx_t = await engine.get('type', idx)
    if not isinstance(idx_t, Int):
        raise MyiaTypeError('Expected Int for index')

    if isinstance(seq_t, Tuple):
        idx_v = await engine.get('value', idx)
        assert idx_v is not ANYTHING
        return seq_t.elements[idx_v]
    elif isinstance(seq_t, List):
        return seq_t.element_type
    else:
        raise MyiaTypeError('Wrong seq type for getitem')


async def infer_type_compare_bin(engine, x, y):
    t = await engine.force_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return Bool()


async def infer_type_arith_bin(engine, x, y):
    t = await engine.force_same('type', x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


for op in [P.add, P.sub, P.mul]:
    type_inferrer_constructors[op] = \
        lambda engine: PrimitiveInferrer(
            engine, 'type', infer_type_arith_bin
        )

for op in [P.lt, P.gt, P.eq, P.le, P.ge]:
    type_inferrer_constructors[op] = \
        lambda engine: PrimitiveInferrer(
            engine, 'type', infer_type_compare_bin
        )
