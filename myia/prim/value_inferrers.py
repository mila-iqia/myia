
import asyncio

from ..infer import ANYTHING, Inferrer, PrimitiveInferrer, GraphInferrer
from ..ir import Graph

from . import ops as P
from .ops import Primitive
from .py_implementations import implementations as pyimpl


class PrimitiveValueInferrer(Inferrer):
    def __init__(self, engine, impl):
        super().__init__(engine)
        self.impl = impl

    async def infer(self, *refs):
        coros = [self.engine.get('value', ref) for ref in refs]
        args = await asyncio.gather(*coros, loop=self.engine.loop)
        if any(arg is ANYTHING for arg in args):
            return ANYTHING
        else:
            return self.impl(*args)


class ValueTrack:

    def __init__(self, implementations, constructors):
        self.implementations = implementations
        self.constructors = constructors

    async def __call__(self, engine, ct):
        v = ct.node.value
        if isinstance(v, Primitive):
            if v in self.constructors:
                return self.constructors[v](engine)
            else:
                return PrimitiveValueInferrer(engine, self.implementations[v])
        elif isinstance(v, Graph):
            return GraphInferrer(engine, 'value', v, ct.context)
        else:
            return v


value_inferrer_constructors = {}
infer_value_constant = ValueTrack(pyimpl, value_inferrer_constructors)


def value_inferrer(prim):
    def deco(fn):
        def constructor(engine):
            return PrimitiveInferrer(engine, 'value', fn)
        value_inferrer_constructors[prim] = constructor
        return fn
    return deco


@value_inferrer(P.if_)
async def infer_value_if(engine, cond, tb, fb):
    v = await engine.get('value', cond)
    if v is True:
        fn = await engine.get('value', tb)
    elif v is False:
        fn = await engine.get('value', fb)
    elif v is ANYTHING:
        return ANYTHING

    return await fn()
