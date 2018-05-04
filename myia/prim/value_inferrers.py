"""Definitions of value inference for primitives."""


import asyncio

from functools import partial

from ..infer import ANYTHING, Inferrer, GraphInferrer, register_inferrer
from ..ir import Graph

from . import ops as P
from .ops import Primitive
from .py_implementations import implementations as pyimpl


class PrimitiveValueInferrer(Inferrer):
    """Infer the return value of a function using its implementation.

    If any input is ANYTHING, the return value will also be ANYTHING.
    The implementation will not be called.
    """

    def __init__(self, engine, prim, impl):
        """Initialize a PrimitiveValueInferrer."""
        super().__init__(engine, prim)
        self.impl = impl

    async def infer(self, *refs):
        """Infer the return value of a function using its implementation."""
        coros = [ref['value'] for ref in refs]
        args = await asyncio.gather(*coros, loop=self.engine.loop)
        if any(arg is ANYTHING for arg in args):
            return ANYTHING
        else:
            return self.impl(*args)


class ValueTrack:
    """Infer the value of a constant.

    Note: the value of a Primitive or of a Graph is an Inferrer.

    Attributes:
        implementations: A map of primitives to implementations.
        constructors: A map of Inferrer constructors. Each constructor
            takes an engine as argument and returns an Inferrer. These
            will be used to infer values for primitives.

    """

    def __init__(self, implementations, constructors):
        """Initialize a ValueTrack."""
        self.implementations = implementations
        self.constructors = constructors

    async def __call__(self, engine, ct):
        """Infer the value of a constant."""
        v = ct.node.value
        if isinstance(v, Primitive):
            if v in self.constructors:
                return self.constructors[v](engine)
            else:
                return PrimitiveValueInferrer(
                    engine, v, self.implementations[v]
                )
        elif isinstance(v, Graph):
            return GraphInferrer(engine, 'value', v, ct.context)
        else:
            return v


# Default constructors
value_inferrer_constructors = {}
infer_value_constant = ValueTrack(pyimpl, value_inferrer_constructors)
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
