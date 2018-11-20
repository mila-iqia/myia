
from .base import (
    ABSENT,
    AbstractBase,
    AbstractValue,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
)

from .inf import (
    AbstractTrack,
    XInferrer,
    GraphXInferrer,
)

from .. import dtype, dshape
from ..infer import ANYTHING
from ..prim import ops as P


abstract_inferrer_constructors = {}


def reg(*prims):
    def deco(cls):
        for prim in prims:
            abstract_inferrer_constructors[prim] = cls
        return cls
    return deco


@reg(P.identity, P.return_)
class IdentityXInferrer(XInferrer):
    async def infer(self, track, x):
        return x


class PrimitiveXInferrer(XInferrer):
    async def infer(self, track, *args):
        rval = await self.infer_structure(track, *args)
        for t in track.subtracks:
            method = getattr(self, f'infer_{t}')
            rval.values[t] = await method(track, *args)
        return rval

    async def infer_structure(self, track, *args):
        raise NotImplementedError()

    async def infer_value(self, track, *args):
        raise NotImplementedError()

    async def infer_type(self, track, *args):
        raise NotImplementedError()

    async def infer_shape(self, track, *args):
        raise NotImplementedError()


class StructuralXInferrer(PrimitiveXInferrer):

    async def infer_value(self, track, *args):
        return ANYTHING

    async def infer_type(self, track, *args):
        return ABSENT

    async def infer_shape(self, track, *args):
        return ABSENT


@reg(P.make_tuple)
class MakeTupleXInferrer(StructuralXInferrer):
    async def infer_structure(self, track, *args):
        return AbstractTuple(args)


@reg(P.tuple_getitem)
class TupleGetitemXInferrer(XInferrer):
    async def infer(self, track, arg, idx):
        i = idx.values['value']
        return arg.elements[i]


class ScalarXInferrer(PrimitiveXInferrer):

    async def infer_structure(self, track, *args):
        assert all(isinstance(arg, AbstractValue) for arg in args)
        return AbstractValue({})

    async def infer_value(self, track, *args):
        values = [arg.values['value'] for arg in args]
        if any(v is ANYTHING for v in values):
            return ANYTHING
        return self.impl(*values)

    async def infer_type(self, track, *args):
        ts = [arg.values['type'] for arg in args]
        assert all(dtype.ismyiatype(t, dtype.Number) for t in ts)
        assert all(t == ts[0] for t in ts)
        return ts[0]

    async def infer_shape(self, track, *args):
        return dshape.NOSHAPE


@reg(P.scalar_add)
class AddXInferrer(ScalarXInferrer):
    def impl(self, x, y):
        return x + y


@reg(P.scalar_mul)
class MulXInferrer(ScalarXInferrer):
    def impl(self, x, y):
        return x * y


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
