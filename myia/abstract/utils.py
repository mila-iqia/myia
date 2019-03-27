"""Utilities for abstract values and inference."""

from functools import reduce
from itertools import chain

from .. import dtype
from ..utils import overload

from .loop import Pending, is_simple, PendingTentative
from .ref import Reference, Context

from .data import (
    ABSENT,
    ANYTHING,
    Possibilities,
    AbstractBase,
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
    VALUE,
    TYPE,
    MyiaTypeError,
    TypeMismatchError,
)


############
# Building #
############


def build_value(a, default=ABSENT):
    """Build a concrete value out of an abstract one.

    A concrete value cannot be built if, for some abstract data, the inferred
    value is ANYTHING. Some types such as AbstractArray cannot be built
    either.

    Arguments:
        a: The abstract value.
        default: A default value to return if the value cannot be built.
            If not provided, a ValueError will be raised in those cases.
    """
    def return_default(err):
        if default is ABSENT:
            raise err
        else:
            return default

    v = a.values.get(VALUE, ABSENT)

    if v is ANYTHING or isinstance(v, Possibilities):
        return return_default(ValueError(v))

    elif v is ABSENT:
        try:
            return _build_value(a)
        except ValueError as e:
            return return_default(e)

    elif isinstance(v, Pending):
        if v.done():
            return v.result()
        else:
            return return_default(ValueError(v))

    else:
        return v


@overload
def _build_value(x: AbstractBase):
    raise ValueError(x)


@overload  # noqa: F811
def _build_value(x: AbstractTuple):
    return tuple(build_value(y) for y in x.elements)


@overload  # noqa: F811
def _build_value(ac: AbstractClass):
    kls = dtype.tag_to_dataclass[ac.tag]
    args = {k: build_value(v) for k, v in ac.attributes.items()}
    return kls(**args)


@overload(bootstrap=True)
def build_type(self, x: AbstractScalar):
    """Build a type from an abstract value."""
    t = x.values[TYPE]
    if isinstance(t, Pending) and t.done():
        t = t.result()
    return t


@overload  # noqa: F811
def build_type(self, x: AbstractFunction):
    return dtype.Function


@overload  # noqa: F811
def build_type(self, x: AbstractTuple):
    return dtype.Tuple[[self(e) for e in x.elements]]


@overload  # noqa: F811
def build_type(self, x: AbstractError):
    return dtype.Problem[x.values[VALUE]]


@overload  # noqa: F811
def build_type(self, x: AbstractList):
    return dtype.List[self(x.element)]


@overload  # noqa: F811
def build_type(self, x: AbstractArray):
    return dtype.Array[self(x.element)]


@overload  # noqa: F811
def build_type(self, x: AbstractClass):
    return dtype.Class[
        x.tag,
        {name: self(x2) for name, x2 in x.attributes.items()},
        x.methods
    ]


@overload  # noqa: F811
def build_type(self, x: AbstractJTagged):
    return dtype.JTagged[self(x.element)]


@overload  # noqa: F811
def build_type(self, x: AbstractType):
    return dtype.TypeType


###########
# Cloning #
###########


@overload(bootstrap=True)
def abstract_clone(self, x: AbstractScalar, *args):
    """Clone an abstract value."""
    return AbstractScalar(self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractFunction, *args):
    return AbstractFunction(value=self(x.get_sync(), *args))


@overload  # noqa: F811
def abstract_clone(self, d: TrackDict, *args):
    return {k: k.clone(v, self) for k, v in d.items()}


@overload  # noqa: F811
def abstract_clone(self, x: AbstractTuple, *args):
    return AbstractTuple(
        [self(y, *args) for y in x.elements],
        self(x.values, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractList, *args):
    return AbstractList(self(x.element, *args), self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractArray, *args):
    return AbstractArray(self(x.element, *args), self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractClass, *args):
    return AbstractClass(
        x.tag,
        {k: self(v, *args) for k, v in x.attributes.items()},
        x.methods,
        self(x.values, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractJTagged, *args):
    return AbstractJTagged(self(x.element, *args))


@overload  # noqa: F811
def abstract_clone(self, x: Pending, *args):
    if x.done():
        return self(x.result(), *args)
    else:
        return x


@overload  # noqa: F811
def abstract_clone(self, x: object, *args):
    return x


#################
# Async cloning #
#################


@overload(bootstrap=True)
async def abstract_clone_async(self, x: AbstractScalar):
    """Clone an abstract value (asynchronous)."""
    return AbstractScalar(await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractFunction):
    return AbstractFunction(*(await self(x.get_sync())))


@overload  # noqa: F811
async def abstract_clone_async(self, d: TrackDict):
    return {k: (await k.async_clone(v, self))
            for k, v in d.items()}


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractTuple):
    return AbstractTuple(
        [(await self(y)) for y in x.elements],
        await self(x.values)
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractList):
    return AbstractList(await self(x.element), await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractArray):
    return AbstractArray(await self(x.element), await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractClass):
    return AbstractClass(
        x.tag,
        {k: (await self(v)) for k, v in x.attributes.items()},
        x.methods,
        await self(x.values)
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: object):
    return x


##############
# Concretize #
##############


@abstract_clone_async.variant
async def concretize_abstract(self, x: Pending):
    """Clone an abstract value while resolving all Pending (asynchronous)."""
    return await self(await x)


@overload  # noqa: F811
async def concretize_abstract(self, r: Reference):
    return Reference(
        r.engine,
        r.node,
        await self(r.context)
    )


@overload  # noqa: F811
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
    """Broaden an abstract value.

    * Concrete values such as 1 or True will be broadened to ANYTHING.
    * Possibilities will be broadened to PendingTentative.

    Arguments:
        d: The abstract data to clone.
        loop: The InferenceLoop, used to broaden Possibilities.
    """
    return {k: k.broaden(v, self, loop) for k, v in d.items()}


@overload  # noqa: F811
def broaden(self, p: Pending, loop):
    return p


@overload  # noqa: F811
def broaden(self, p: Possibilities, loop):
    if loop is None:
        return p
    else:
        # Broadening Possibilities creates a PendingTentative. This allows
        # us to avoid resolving them earlier than we would like.
        return loop.create_pending_tentative(tentative=p)


###############
# Sensitivity #
###############


@abstract_clone.variant
def sensitivity_transform(self, x: AbstractFunction):
    """Return an abstract value for the sensitivity of x.

    * The sensitivity of a function is an Env
    * The sensitivity of J(x) is x
    """
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
    })


@overload  # noqa: F811
def sensitivity_transform(self, x: AbstractJTagged):
    return self(x.element)


#########
# Merge #
#########


@overload.wrapper(bootstrap=False)
def amerge(__call__, x1, x2, loop, forced, accept_pending=True):
    """Merge two values.

    If forced is False, amerge will return a superset of x1 and x2, if it
    exists.

    If the forced argument is True, amerge will either return x1 or fail.
    This makes a difference in some situations:

        * amerge(1, 2, forced=False) => ANYTHING
        * amerge(1, 2, forced=True) => Error
        * amerge(ANYTHING, 1234, forced=True) => ANYTHING
        * amerge(1234, ANYTHING, forced=True) => Error

    Arguments:
        x1: The first value to merge
        x2: The second value to merge
        loop: The InferenceLoop
        forced: Whether we are already committed to returning x1 or not.
        accept_pending: Whether we accept to merge two Pending, unresolved
            values.
    """
    isp1 = isinstance(x1, Pending)
    isp2 = isinstance(x2, Pending)
    if isp1 and x1.done() and not forced:
        # TODO: Check if the case when forced is True is sound
        x1 = x1.result()
        isp1 = False
    if isp2 and x2.done():
        x2 = x2.result()
        isp2 = False
    if isinstance(x1, PendingTentative):
        assert not x1.done()  # TODO: handle this case?
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
        return x1
    elif x2 is ANYTHING:
        if forced:
            raise TypeMismatchError(x1, x2)
        return x2
    elif type(x1) is not type(x2):
        raise MyiaTypeError(
            f'Type mismatch: {type(x1)} != {type(x2)}; {x1} != {x2}'
        )
    else:
        return __call__(x1, x2, loop, forced)


@overload  # noqa: F811
def amerge(x1: Possibilities, x2, loop, forced):
    if x1.issuperset(x2):
        return x1
    if forced:
        raise MyiaTypeError(
            'Additional possibilities cannot be merged.'
        )
    else:
        return Possibilities(x1 | x2)


@overload  # noqa: F811
def amerge(x1: dtype.TypeMeta, x2, loop, forced):
    if x1 != x2:
        raise TypeMismatchError(x1, x2)
    return x1


@overload  # noqa: F811
def amerge(x1: (dict, TrackDict), x2, loop, forced):
    if set(x1.keys()) != set(x2.keys()):
        # This shouldn't be possible at the moment
        raise AssertionError(f'Keys mismatch')
    changes = False
    rval = type(x1)()
    for k, v in x1.items():
        res = amerge(v, x2[k], loop, forced)
        if res is not v:
            changes = True
        rval[k] = res
    return x1 if forced or not changes else rval


@overload  # noqa: F811
def amerge(x1: tuple, x2, loop, forced):
    if len(x1) != len(x2):  # pragma: no cover
        raise MyiaTypeError(f'Tuple length mismatch')
    changes = False
    rval = []
    for v1, v2 in zip(x1, x2):
        res = amerge(v1, v2, loop, forced)
        if res is not v1:
            changes = True
        rval.append(res)
    return x1 if forced or not changes else tuple(rval)


@overload  # noqa: F811
def amerge(x1: AbstractScalar, x2, loop, forced):
    values = amerge(x1.values, x2.values, loop, forced)
    if forced or values is x1.values:
        return x1
    return AbstractScalar(values)


@overload  # noqa: F811
def amerge(x1: AbstractFunction, x2, loop, forced):
    values = amerge(x1.get_sync(), x2.get_sync(), loop, forced)
    if forced or values is x1.values:
        return x1
    return AbstractFunction(*values)


@overload  # noqa: F811
def amerge(x1: AbstractTuple, x2, loop, forced):
    args1 = (x1.elements, x1.values)
    args2 = (x2.elements, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractTuple(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractArray, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractArray(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractList, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractList(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractClass, x2, loop, forced):
    args1 = (x1.tag, x1.attributes, x1.methods, x1.values)
    args2 = (x2.tag, x2.attributes, x2.methods, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractClass(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractJTagged, x2, loop, forced):
    args1 = x1.element
    args2 = x2.element
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractJTagged(merged)


@overload  # noqa: F811
def amerge(x1: (int, float, bool), x2, loop, forced):
    if forced and x1 != x2:
        raise TypeMismatchError(x1, x2)
    return x1 if x1 == x2 else ANYTHING


@overload  # noqa: F811
def amerge(x1: object, x2, loop, forced):
    if x1 != x2:
        raise TypeMismatchError(x1, x2)
    return x1


def bind(loop, committed, resolved, pending):
    """Bind Pendings together.

    Arguments:
        loop: The InferenceLoop.
        committed: Either None, or an abstract value that we are already
            committed to, which will force the merge to return that value.
        resolved: A set of Pendings that have already been resolved.
        pending: A set of unresolved Pendings.
    """
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
                rval.set_result(v)

    for p in pending:
        p.add_done_callback(resolve)

    def premature_resolve():
        # This is what force_resolve() on the result will do
        nonlocal committed
        # We merge what we have so far
        committed = amergeall()
        # We broaden the result so that the as-of-yet unresolved stuff
        # can be merged more easily.
        committed = broaden(committed, loop)
        resolved.clear()
        return committed

    def priority():
        # Cannot force resolve unless we have at least one resolved Pending
        if not resolved and committed is None:
            return None
        if any(is_simple(x) for x in chain([committed], resolved, pending)):
            return 1000
        return -1000

    if any(is_simple(x) for x in chain(resolved, pending)):
        # rval = None because we will not make a new Pending
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
            raise AssertionError('unreachable')

    else:
        rval = loop.create_pending(
            resolve=premature_resolve,
            priority=priority,
        )
        rval.equiv.update(resolved)
        for p in pending:
            rval.tie(p)

        return rval
