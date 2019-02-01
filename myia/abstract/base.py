
import math
import numpy
from dataclasses import dataclass, is_dataclass
from functools import reduce
from itertools import chain
from typing import Tuple

from .. import dtype
from ..ir import Graph
from ..prim import Primitive
from ..debug.utils import mixin
from ..utils import overload, UNKNOWN, Named

from .loop import Pending, is_simple, PendingTentative
from .ref import Reference, Context

from .data import (  # noqa
    ABSENT,
    ANYTHING,
    VOID,
    DEAD,
    POLY,
    INACCESSIBLE,
    Possibilities,
    TrackableFunction,
    GraphAndContext,
    PartialApplication,
    JTransformedFunction,
    VirtualFunction,
    TypedPrimitive,
    DummyFunction,
    AbstractBase,
    AbstractValue,
    AbstractScalar,
    AbstractType,
    AbstractError,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    AbstractJTagged,
    TrackDict,
    Subtrack,
    VALUE,
    TYPE,
    SHAPE,
    InferenceError,
    MyiaTypeError,
)


############
# Building #
############


@overload(bootstrap=True)
def to_value(self, x: AbstractScalar, *args):
    return x.values[VALUE]


@overload
def to_value(self, x: AbstractFunction, *args):
    return x.values[VALUE]


@overload
def to_value(self, x: AbstractTuple, *args):
    return tuple(self(y, *args) for y in x.elements)


@overload
def to_value(self, x: AbstractError, *args):
    raise ValueError('Cannot build error.')


@overload
def to_value(self, x: AbstractList, *args):
    raise ValueError('Cannot build list.')


@overload
def to_value(self, x: AbstractArray, *args):
    raise ValueError('Cannot build array.')


@overload
def to_value(self, x: AbstractClass, *args):
    raise ValueError('Cannot build struct.')


@overload
def to_value(self, x: AbstractJTagged, *args):
    raise ValueError('Cannot build jtagged.')


@overload
def to_value(self, x: AbstractType, *args):
    raise ValueError('Cannot build type.')


###########
# Cloning #
###########


@overload(bootstrap=True)
def abstract_clone(self, x: AbstractScalar, *args):
    return AbstractScalar(self(x.values, *args))


@overload
def abstract_clone(self, x: AbstractFunction, *args):
    return AbstractFunction(value=self(x.values[VALUE], *args))


@overload
def abstract_clone(self, d: TrackDict, *args):
    return {k: k.clone(v, self) for k, v in d.items()}


@overload
def abstract_clone(self, x: AbstractTuple, *args):
    return AbstractTuple(
        [self(y, *args) for y in x.elements],
        self(x.values, *args)
    )


@overload
def abstract_clone(self, x: AbstractList, *args):
    return AbstractList(self(x.element, *args), self(x.values, *args))


@overload
def abstract_clone(self, x: AbstractArray, *args):
    return AbstractArray(self(x.element, *args), self(x.values, *args))


@overload
def abstract_clone(self, x: AbstractClass, *args):
    return AbstractClass(
        x.tag,
        {k: self(v, *args) for k, v in x.attributes.items()},
        x.methods,
        self(x.values, *args)
    )


@overload
def abstract_clone(self, x: AbstractJTagged, *args):
    return AbstractJTagged(self(x.element, *args))


@overload
def abstract_clone(self, x: object, *args):
    return x


#################
# Async cloning #
#################


@overload(bootstrap=True)
async def abstract_clone_async(self, x: AbstractScalar):
    return AbstractScalar(await self(x.values))


@overload
async def abstract_clone_async(self, x: AbstractFunction):
    return AbstractFunction(*(await self(x.values[VALUE])))


@overload
async def abstract_clone_async(self, d: TrackDict):
    return {k: (await k.async_clone(v, self))
            for k, v in d.items()}


@overload
async def abstract_clone_async(self, x: AbstractTuple):
    return AbstractTuple(
        [(await self(y)) for y in x.elements],
        await self(x.values)
    )


@overload
async def abstract_clone_async(self, x: AbstractList):
    return AbstractList(await self(x.element), await self(x.values))


@overload
async def abstract_clone_async(self, x: AbstractArray):
    return AbstractArray(await self(x.element), await self(x.values))


@overload
async def abstract_clone_async(self, x: AbstractClass):
    return AbstractClass(
        x.tag,
        {k: (await self(v)) for k, v in x.attributes.items()},
        x.methods,
        await self(x.values)
    )


@overload
async def abstract_clone_async(self, x: object):
    return x


##############
# Concretize #
##############


@abstract_clone_async.variant
async def concretize_abstract(self, x: Pending):
    return await self(await x)


@overload
async def concretize_abstract(self, r: Reference):
    return Reference(
        r.engine,
        r.node,
        await self(r.context)
    )


@overload
async def concretize_abstract(self, ctx: Context):
    c_argkey = [await self(x) for x in ctx.argkey]
    return Context(
        await self(ctx.parent),
        ctx.graph,
        tuple(c_argkey)
    )


###########
# Broaden #
###########


@abstract_clone.variant
def broaden(self, d: TrackDict, loop):
    return {k: k.broaden(v, self, loop) for k, v in d.items()}


@overload
def broaden(self, p: Pending, loop):
    return p


@overload
def broaden(self, p: Possibilities, loop):
    if loop is None:
        return p
    else:
        return loop.create_pending_tentative(tentative=p)


###############
# Sensitivity #
###############


@abstract_clone.variant
def sensitivity_transform(self, x: AbstractFunction):
    v = x.values[VALUE]
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
    })


@overload
def sensitivity_transform(self, x: AbstractJTagged):
    return self(x.element)


#########
# Merge #
#########


@overload(bootstrap=True)
def _amerge(self, x1: Possibilities, x2, loop, forced):
    if x1.issuperset(x2):
        return x1
    if forced:
        raise MyiaTypeError('Cannot merge Possibilities')
    else:
        return Possibilities(x1 | x2)


@overload
def _amerge(self, x1: dtype.TypeMeta, x2, loop, forced):
    if x1 != x2:
        raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    return x1


@overload
def _amerge(self, x1: (dict, TrackDict), x2, loop, forced):
    if set(x1.keys()) != set(x2.keys()):
        raise MyiaTypeError(f'Keys mismatch')
    changes = False
    rval = type(x1)()
    for k, v in x1.items():
        if isinstance(k, Subtrack) and not k.eq_relevant():
            continue
        res = amerge(v, x2[k], loop, forced)
        if res is not v:
            changes = True
        rval[k] = res
    return x1 if forced or not changes else rval


@overload
def _amerge(self, x1: tuple, x2, loop, forced):
    if len(x1) != len(x2):
        raise MyiaTypeError(f'Tuple length mismatch')
    changes = False
    rval = []
    for v1, v2 in zip(x1, x2):
        res = amerge(v1, v2, loop, forced)
        if res is not v1:
            changes = True
        rval.append(res)
    return x1 if forced or not changes else tuple(rval)


@overload
def _amerge(self, x1: AbstractScalar, x2, loop, forced):
    values = amerge(x1.values, x2.values, loop, forced)
    if forced or values is x1.values:
        return x1
    return AbstractScalar(values)


@overload
def _amerge(self, x1: AbstractFunction, x2, loop, forced):
    values = amerge(x1.values[VALUE], x2.values[VALUE], loop, forced)
    if forced or values is x1.values:
        return x1
    return AbstractFunction(*values)


@overload
def _amerge(self, x1: AbstractTuple, x2, loop, forced):
    args1 = (x1.elements, x1.values)
    args2 = (x2.elements, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractTuple(*merged)


@overload
def _amerge(self, x1: AbstractArray, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractArray(*merged)


@overload
def _amerge(self, x1: AbstractList, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractList(*merged)


@overload
def _amerge(self, x1: AbstractClass, x2, loop, forced):
    args1 = (x1.tag, x1.attributes, x1.methods, x1.values)
    args2 = (x2.tag, x2.attributes, x2.methods, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractClass(*merged)


@overload
def _amerge(self, x1: AbstractJTagged, x2, loop, forced):
    args1 = x1.element
    args2 = x2.element
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractJTagged(merged)


@overload
def _amerge(self, x1: (int, float, bool), x2, loop, forced):
    if x1 is ANYTHING:
        return x1
    if forced:
        if x1 != x2:
            raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    elif x2 is ANYTHING or x1 != x2:
        return ANYTHING
    return x1


@overload
def _amerge(self, x1: Named, x2, loop, forced):
    if x1 is ANYTHING:
        if x2 is ANYTHING:
            return x1
        return self[type(x2)](x1, x2, loop, forced)
    if x1 != x2:
        raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    return x1


@overload
def _amerge(self, x1: object, x2, loop, forced):
    if x1 is ANYTHING:
        return ANYTHING
    elif x2 is ANYTHING and not forced:
        return ANYTHING
    if x1 != x2:
        raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    return x1


def amerge(x1, x2, loop, forced, accept_pending=True):
    isp1 = isinstance(x1, Pending)
    isp2 = isinstance(x2, Pending)
    if isp1 and x1.done():
        assert forced is False  # TODO: fix this?
        x1 = x1.result()
        isp1 = False
    if isp2 and x2.done():
        x2 = x2.result()
        isp2 = False
    if isinstance(x1, PendingTentative):
        assert not x1.done()  # TODO: fix this?
        x1.tentative = amerge(x1.tentative, x2, loop, False, True)
        return x1
    if (isp1 or isp2) and not accept_pending:
        raise AssertionError('Cannot have Pending here.')
    if isp1 and isp2:
        return bind(loop, x1 if forced else None, [], [x1, x2])
    elif isp1:
        return bind(loop, x1 if forced else None, [x2], [x1])
    elif isp2:
        return bind(loop, x1 if forced else None, [x1], [x2])
    elif x1 is ANYTHING:
        if x2 is ANYTHING:
            return x1
        return _amerge[type(x2)](x1, x2, loop, forced)
    elif x2 is not ANYTHING and type(x1) is not type(x2):
        raise MyiaTypeError(
            f'Type mismatch: {type(x1)} != {type(x2)}; {x1} != {x2}'
        )
    else:
        return _amerge(x1, x2, loop, forced)


def bind(loop, committed, resolved, pending):

    def amergeall():
        if committed is None:
            v = reduce(lambda x1, x2: amerge(x1, x2,
                                             loop=loop,
                                             forced=False,
                                             accept_pending=False),
                       resolved)
        else:
            v = reduce(lambda x1, x2: amerge(x1, x2,
                                             loop=loop,
                                             forced=True,
                                             accept_pending=False),
                       resolved, committed)
        return v

    resolved = list(resolved)
    pending = set(pending)
    assert pending

    def resolve(fut):
        nonlocal committed
        pending.remove(fut)
        result = fut.result()
        if fut is committed:
            committed = result
        resolved.append(result)
        if not pending:
            v = amergeall()
            if rval is not None and not rval.done():
                rval.resolve_to(v)

    for p in pending:
        p.add_done_callback(resolve)

    def premature_resolve():
        nonlocal committed
        committed = amergeall()
        committed = broaden(committed, loop)
        resolved.clear()
        return committed

    def priority():
        if not resolved and committed is None:
            return None
        if any(is_simple(x) for x in chain([committed], resolved, pending)):
            return 1000
        return prio

    if any(is_simple(x) for x in chain(resolved, pending)):
        rval = None

        if pending:
            p, *rest = pending
            p.equiv.update(resolved)
            for p2 in rest:
                p.tie(p2)

        if resolved:
            return resolved[0]
        else:
            for p in pending:
                if is_simple(p):
                    return p
            return p

    else:
        prio = (-1000
                if any(p.priority() is None for p in pending)
                else min(p.priority() for p in pending) - 1)
        rval = loop.create_pending(
            resolve=premature_resolve,
            priority=priority,
        )
        rval.equiv.update(resolved)
        for p in pending:
            rval.tie(p)

        return rval
