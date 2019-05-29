"""Utilities for abstract values and inference."""

from dataclasses import dataclass
import typing
import numpy as np
from functools import reduce
from itertools import chain
from types import GeneratorType, AsyncGeneratorType

from .. import dtype
from ..utils import overload, is_dataclass_type, dataclass_methods, intern

from .loop import Pending, is_simple, PendingTentative, \
    find_coherent_result_sync

from .data import (
    ABSENT,
    ANYTHING,
    Possibilities,
    AbstractAtom,
    AbstractStructure,
    AbstractValue,
    AbstractScalar,
    AbstractError,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    AbstractADT,
    AbstractJTagged,
    AbstractUnion,
    abstract_union,
    TrackDict,
    PartialApplication,
    JTransformedFunction,
    VALUE,
    TYPE,
    SHAPE,
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
def _build_value(x: AbstractValue):
    raise ValueError(x)


@overload  # noqa: F811
def _build_value(x: AbstractTuple):
    return tuple(build_value(y) for y in x.elements)


@overload  # noqa: F811
def _build_value(ac: AbstractClass):
    args = {k: build_value(v) for k, v in ac.attributes.items()}
    return ac.tag(**args)


_default_type_params = {
    tuple: (),
    list: (object,),
}


@overload(bootstrap=True)
def type_to_abstract(self, t: dtype.TypeMeta):
    """Convert a type to an AbstractValue."""
    return self[t](t)


@overload  # noqa: F811
def type_to_abstract(self, t: AbstractValue):
    return t


@overload  # noqa: F811
def type_to_abstract(self, t: (dtype.Number, dtype.Bool, dtype.EnvType,
                               dtype.SymbolicKeyType, dtype.Nil)):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: t,
    })


@overload  # noqa: F811
def type_to_abstract(self, t: type):
    if is_dataclass_type(t):
        fields = t.__dataclass_fields__
        attributes = {name: ANYTHING
                      if isinstance(field.type, (str, type(None)))
                      else self(field.type)
                      for name, field in fields.items()}
        return AbstractClass(t, attributes, dataclass_methods(t))

    elif t is object:
        return ANYTHING

    else:
        return _t2a_helper[t](t, _default_type_params.get(t, None))


@overload  # noqa: F811
def type_to_abstract(self, t: typing._GenericAlias):
    args = tuple(object if isinstance(arg, typing.TypeVar) else arg
                 for arg in t.__args__)
    return _t2a_helper[t.__origin__](t, args)


@overload  # noqa: F811
def type_to_abstract(self, t: object):
    raise MyiaTypeError(f'{t} is not a recognized type')


@overload
def _t2a_helper(main: tuple, args):
    if args == () or args is None:
        targs = ANYTHING
    elif args == ((),):
        targs = []
    else:
        targs = [type_to_abstract(a) for a in args]
    return AbstractTuple(targs)


@overload  # noqa: F811
def _t2a_helper(main: list, args):
    arg, = args
    return AbstractList(type_to_abstract(arg))


@overload  # noqa: F811
def _t2a_helper(main: np.ndarray, args):
    arg, = args
    arg = type_to_abstract(arg)
    shp = ANYTHING
    return AbstractArray(arg, {SHAPE: shp})


@overload  # noqa: F811
def _t2a_helper(main: int, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
    })


@overload  # noqa: F811
def _t2a_helper(main: float, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Float[64],
    })


@overload  # noqa: F811
def _t2a_helper(main: bool, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Bool,
    })


@overload  # noqa: F811
def _t2a_helper(main: AbstractFunction, args):
    return AbstractFunction(value=ANYTHING)


@overload  # noqa: F811
def _t2a_helper(main: AbstractError, args):
    return AbstractError(ANYTHING)


def type_token(x):
    """Build a type from an abstract value."""
    if isinstance(x, AbstractScalar):
        return x.dtype()
    else:
        return type(x)


############
# Checking #
############


@dataclass
class CheckState:
    """State of abstract_check."""

    cache: dict
    prop: str


@overload.wrapper(
    initial_state=lambda: CheckState({}, None)
)
def abstract_check(__call__, self, x, *args):
    """Check that a predicate applies to a given object."""
    def proceed():
        if prop:
            if hasattr(x, prop):
                return getattr(x, prop) is x
            elif __call__(self, x, *args):
                if isinstance(x, AbstractValue):
                    setattr(x, prop, x)
                return True
            else:
                return False
        else:
            return __call__(self, x, *args)

    prop = self.state.prop
    cache = self.state.cache

    try:
        rval = cache.get(x, None)
    except TypeError:
        return proceed()

    if rval is None:
        cache[x] = True
        cache[x] = proceed()
        return cache[x]
    else:
        return rval


@overload  # noqa: F811
def abstract_check(self, x: TrackDict, *args):
    return all(self(v, *args) for v in x.values())


@overload  # noqa: F811
def abstract_check(self, x: AbstractScalar, *args):
    return self(x.values, *args)


@overload  # noqa: F811
def abstract_check(self, xs: AbstractStructure, *args):
    return all(self(x, *args) for x in xs.children())


@overload  # noqa: F811
def abstract_check(self, xs: AbstractAtom, *args):
    return True


@overload  # noqa: F811
def abstract_check(self, x: AbstractFunction, *args):
    return self(x.values, *args)


@overload  # noqa: F811
def abstract_check(self, x: Possibilities, *args):
    return all(self(v, *args) for v in x)


@overload  # noqa: F811
def abstract_check(self, t: PartialApplication, *args):
    return self(t.fn, *args) and all(self(v, *args) for v in t.args)


@overload  # noqa: F811
def abstract_check(self, t: JTransformedFunction, *args):
    return self(t.fn, *args)


@overload  # noqa: F811
def abstract_check(self, xs: object, *args):
    return True


###########
# Cloning #
###########


@dataclass
class CloneState:
    """State of abstract_clone."""

    cache: dict
    prop: str
    check: callable


def _make_constructor(inst):
    def f(*args, **kwargs):
        inst.__init__(*args, **kwargs)
        return inst
    return f


@overload.wrapper(
    initial_state=lambda: CloneState({}, None, None),
    postprocess=intern,
)
def abstract_clone(__call__, self, x, *args):
    """Clone an abstract value."""
    def proceed():
        if isinstance(x, AbstractValue) and x in cache:
            return cache[x]
        result = __call__(self, x, *args)
        if not isinstance(result, GeneratorType):
            return result
        cls = result.send(None)
        inst = cls.empty()
        cache[x] = inst
        constructor = _make_constructor(inst)
        try:
            result.send(constructor)
        except StopIteration as e:
            assert e.value is inst
            return inst
        else:
            raise AssertionError(
                'Generators in abstract_clone must yield once, then return.'
            )

    cache = self.state.cache
    prop = self.state.prop
    if prop:
        if hasattr(x, prop):
            return getattr(x, prop)
        elif isinstance(x, AbstractValue):
            if self.state.check(x, *args):
                res = x
            else:
                res = proceed()
            setattr(x, prop, res)
            return res
        else:
            return proceed()
    else:
        return proceed()


@overload  # noqa: F811
def abstract_clone(self, x: AbstractScalar, *args):
    return AbstractScalar(self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractFunction, *args):
    return (yield AbstractFunction)(value=self(x.get_sync(), *args))


@overload  # noqa: F811
def abstract_clone(self, d: TrackDict, *args):
    return {k: k.clone(v, self) for k, v in d.items()}


@overload  # noqa: F811
def abstract_clone(self, x: AbstractTuple, *args):
    return (yield AbstractTuple)(
        [self(y, *args) for y in x.elements],
        self(x.values, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractList, *args):
    return (yield AbstractList)(self(x.element, *args), self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractArray, *args):
    return (yield AbstractArray)(self(x.element, *args), self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractClass, *args):
    return (yield AbstractClass)(
        x.tag,
        {k: self(v, *args) for k, v in x.attributes.items()},
        x.methods,
        self(x.values, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractADT, *args):
    return (yield AbstractADT)(
        x.tag,
        {k: self(v, *args) for k, v in x.attributes.items()},
        x.methods,
        self(x.values, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractUnion, *args):
    return (yield AbstractUnion)([self(y, *args) for y in x.options])


@overload  # noqa: F811
def abstract_clone(self, x: AbstractJTagged, *args):
    return (yield AbstractJTagged)(self(x.element, *args))


@overload  # noqa: F811
def abstract_clone(self, x: Possibilities, *args):
    return Possibilities([self(v, *args) for v in x])


@overload  # noqa: F811
def abstract_clone(self, x: PartialApplication, *args):
    return PartialApplication(
        self(x.fn, *args),
        [self(arg, *args) for arg in x.args]
    )


@overload  # noqa: F811
def abstract_clone(self, x: JTransformedFunction, *args):
    return JTransformedFunction(self(x.fn, *args))


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


async def _abstract_clone_async_post(x):
    return intern(await x)


@overload.wrapper(
    initial_state=lambda: CloneState({}, None, None),
    postprocess=_abstract_clone_async_post,
)
async def abstract_clone_async(__call__, self, x, *args):
    """Clone an abstract value (asynchronous)."""
    async def proceed():
        if isinstance(x, AbstractValue) and x in cache:
            return cache[x]

        call = __call__(self, x, *args)
        if isinstance(call, AsyncGeneratorType):
            cls = await call.asend(None)
            inst = cls.empty()
            cache[x] = inst
            constructor = _make_constructor(inst)
            rval = await call.asend(constructor)
            assert rval is inst
            return rval
        else:
            return await call

    cache = self.state.cache
    prop = self.state.prop
    if prop:
        if hasattr(x, prop):
            return getattr(x, prop)
        elif isinstance(x, AbstractValue):
            if self.state.check(x, *args):
                res = x
            else:
                res = await proceed()
            setattr(x, prop, res)
            return res
        else:
            return await proceed()
    else:
        return await proceed()


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractScalar):
    return AbstractScalar(await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractFunction):
    yield (yield AbstractFunction)(*(await self(x.get_sync())))


@overload  # noqa: F811
async def abstract_clone_async(self, d: TrackDict):
    return {k: (await k.async_clone(v, self))
            for k, v in d.items()}


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractTuple):
    yield (yield AbstractTuple)(
        [(await self(y)) for y in x.elements],
        await self(x.values)
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractList):
    yield (yield AbstractList)(await self(x.element), await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractArray):
    yield (yield AbstractArray)(await self(x.element), await self(x.values))


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractClass):
    yield (yield AbstractClass)(
        x.tag,
        {k: (await self(v)) for k, v in x.attributes.items()},
        x.methods,
        await self(x.values)
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractADT):
    yield (yield AbstractADT)(
        x.tag,
        {k: (await self(v)) for k, v in x.attributes.items()},
        x.methods,
        await self(x.values)
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: AbstractUnion):
    yield (yield AbstractUnion)([await self(y) for y in x.options])


@overload  # noqa: F811
async def abstract_clone_async(self, x: Possibilities):
    return Possibilities([await self(v) for v in x])


@overload  # noqa: F811
async def abstract_clone_async(self, x: PartialApplication):
    return PartialApplication(
        await self(x.fn),
        [await self(arg) for arg in x.args]
    )


@overload  # noqa: F811
async def abstract_clone_async(self, x: object):
    return x


##################
# Concrete check #
##################


@abstract_check.variant(
    initial_state=lambda: CheckState(cache={}, prop='_concrete')
)
def _is_concrete(self, x: Pending):
    return False


##############
# Concretize #
##############


@abstract_clone_async.variant(
    initial_state=lambda: CloneState({}, '_concrete', _is_concrete)
)
async def concretize_abstract(self, x: Pending):
    """Clone an abstract value while resolving all Pending (asynchronous)."""
    return await self(await x)


###############
# Broad check #
###############


@abstract_check.variant(
    initial_state=lambda: CheckState(cache={}, prop='_broad')
)
def _is_broad(self, x: object, loop):
    return x is ANYTHING


@overload  # noqa: F811
def _is_broad(self, x: (AbstractScalar, AbstractFunction), loop):
    return self(x.values[VALUE], loop)


@overload  # noqa: F811
def _is_broad(self, x: Possibilities, loop):
    if loop is None:
        return all(self(v, loop) for v in x)
    else:
        return False


###########
# Broaden #
###########


@abstract_clone.variant(
    initial_state=lambda: CloneState({}, '_broad', _is_broad)
)
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
def broaden(self, p: Possibilities, loop):
    if loop is None:
        return Possibilities([self(v, loop) for v in p])
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
def amerge(__call__, x1, x2, loop, forced, bind_pending=True,
           accept_pending=True):
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
        bind_pending: Whether we bind two Pending, unresolved values.
        accept_pending: Works the same as bind_pending, but only for the
            top level call.
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
    if (isp1 or isp2) and (not accept_pending or not bind_pending):
        if forced and isp1:
            raise MyiaTypeError('Cannot have Pending here.')
        if isp1:
            def chk(a):
                return amerge(a, x2, loop, forced, bind_pending)
            return find_coherent_result_sync(x1, chk)
        if isp2:
            def chk(a):
                return amerge(x1, a, loop, forced, bind_pending)
            return find_coherent_result_sync(x2, chk)
    if isinstance(x1, PendingTentative):
        assert not x1.done()  # TODO: handle this case?
        x1.tentative = amerge(x1.tentative, x2, loop, False, True,
                              bind_pending)
        return x1
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
    elif type(x1) is not type(x2) and not isinstance(x1, (int, float, bool)):
        raise MyiaTypeError(
            f'Type mismatch: {type(x1)} != {type(x2)}; {x1} != {x2}'
        )
    else:
        return __call__(x1, x2, loop, forced, bind_pending)


@overload  # noqa: F811
def amerge(x1: Possibilities, x2, loop, forced, bp):
    if set(x1).issuperset(set(x2)):
        return x1
    if forced:
        raise MyiaTypeError(
            'Additional possibilities cannot be merged.'
        )
    else:
        return Possibilities(x1 + x2)


@overload  # noqa: F811
def amerge(x1: dtype.TypeMeta, x2, loop, forced, bp):
    if issubclass(x2, x1):
        return x1
    elif not forced and issubclass(x1, x2):
        return x2
    else:
        raise TypeMismatchError(x1, x2)


@overload  # noqa: F811
def amerge(x1: (dict, TrackDict), x2, loop, forced, bp):
    if set(x1.keys()) != set(x2.keys()):
        # This shouldn't be possible at the moment
        raise AssertionError(f'Keys mismatch')
    changes = False
    rval = type(x1)()
    for k, v in x1.items():
        res = amerge(v, x2[k], loop, forced, bp)
        if res is not v:
            changes = True
        rval[k] = res
    return x1 if forced or not changes else rval


@overload  # noqa: F811
def amerge(x1: (tuple, list), x2, loop, forced, bp):
    if len(x1) != len(x2):  # pragma: no cover
        raise MyiaTypeError(f'Tuple length mismatch')
    changes = False
    rval = []
    for v1, v2 in zip(x1, x2):
        res = amerge(v1, v2, loop, forced, bp)
        if res is not v1:
            changes = True
        rval.append(res)
    return x1 if forced or not changes else type(x1)(rval)


@overload  # noqa: F811
def amerge(x1: AbstractScalar, x2, loop, forced, bp):
    values = amerge(x1.values, x2.values, loop, forced, bp)
    if forced or values is x1.values:
        return x1
    return AbstractScalar(values)


@overload  # noqa: F811
def amerge(x1: AbstractFunction, x2, loop, forced, bp):
    values = amerge(x1.get_sync(), x2.get_sync(), loop, forced, bp)
    if forced or values is x1.values:
        return x1
    return AbstractFunction(*values)


@overload  # noqa: F811
def amerge(x1: AbstractTuple, x2, loop, forced, bp):
    args1 = (x1.elements, x1.values)
    args2 = (x2.elements, x2.values)
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractTuple(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractArray, x2, loop, forced, bp):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractArray(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractList, x2, loop, forced, bp):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractList(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractClass, x2, loop, forced, bp):
    args1 = (x1.tag, x1.attributes, x1.methods, x1.values)
    args2 = (x2.tag, x2.attributes, x2.methods, x2.values)
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractClass(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractADT, x2, loop, forced, bp):
    args1 = (x1.tag, x1.attributes, x1.methods, x1.values)
    args2 = (x2.tag, x2.attributes, x2.methods, x2.values)
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractADT(*merged)


@overload  # noqa: F811
def amerge(x1: AbstractJTagged, x2, loop, forced, bp):
    args1 = x1.element
    args2 = x2.element
    merged = amerge(args1, args2, loop, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractJTagged(merged)


@overload  # noqa: F811
def amerge(x1: (int, float, bool), x2, loop, forced, bp):
    if forced and x1 != x2:
        raise TypeMismatchError(x1, x2)
    return x1 if x1 == x2 else ANYTHING


@overload  # noqa: F811
def amerge(x1: object, x2, loop, forced, bp):
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


###########################
# Typing-related routines #
###########################


def typecheck(model, abstract):
    """Check that abstract matches the model."""
    try:
        amerge(model, abstract, forced=True, bind_pending=False, loop=None)
    except MyiaTypeError:
        return False
    else:
        return True


def split_type(t, model):
    """Checks t against the model and return matching/non-matching subtypes.

    * If t is a Union, return a Union that fully matches model, and a Union
      that does not match model. No matches in either case returns None for
      that case.
    * Otherwise, return (t, None) or (None, t) depending on whether t matches
      the model.
    """
    if isinstance(t, AbstractUnion):
        matching = [(opt, typecheck(model, opt))
                    for opt in set(t.options)]
        t1 = abstract_union([opt for opt, m in matching if m])
        t2 = abstract_union([opt for opt, m in matching if not m])
        return t1, t2
    elif typecheck(model, t):
        return t, None
    else:
        return None, t


def hastype_helper(value, model):
    """Helper to implement hastype."""
    match, nomatch = split_type(value, model)
    if match is None:
        return False
    elif nomatch is None:
        return True
    else:
        return ANYTHING
