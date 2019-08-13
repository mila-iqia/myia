"""Utilities for abstract values and inference."""

import typing
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import AsyncGeneratorType, GeneratorType

import numpy as np

from .. import dtype
from ..utils import (
    ADT,
    Cons,
    Empty,
    MyiaTypeError,
    TypeMismatchError,
    dataclass_methods,
    intern,
    is_dataclass_type,
    overload,
)
from .data import (
    ABSENT,
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractADT,
    AbstractArray,
    AbstractAtom,
    AbstractBottom,
    AbstractClass,
    AbstractClassBase,
    AbstractDict,
    AbstractError,
    AbstractFunction,
    AbstractJTagged,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractStructure,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    AbstractValue,
    JTransformedFunction,
    PartialApplication,
    Possibilities,
    TaggedPossibilities,
    TrackDict,
)
from .loop import (
    Pending,
    PendingTentative,
    find_coherent_result_sync,
    is_simple,
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
    return ac.constructor(**args)


_default_type_params = {
    tuple: (),
    list: (object,),
}


@overload(bootstrap=True)
def type_to_abstract(self, t: dtype.TypeMeta):
    """Convert a type to an AbstractValue.

    If the value is already an AbstractValue, returns it directly.
    """
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
        if issubclass(t, ADT):
            return AbstractADT(t, attributes, dataclass_methods(t))
        else:
            return AbstractClass(t, attributes, dataclass_methods(t))

    elif t is object:
        return ANYTHING

    else:
        return pytype_to_abstract[t](t, _default_type_params.get(t, None))


@overload  # noqa: F811
def type_to_abstract(self, t: typing._GenericAlias):
    args = tuple(object if isinstance(arg, typing.TypeVar) else arg
                 for arg in t.__args__)
    return pytype_to_abstract[t.__origin__](t, args)


@overload  # noqa: F811
def type_to_abstract(self, t: object):
    raise MyiaTypeError(f'{t} is not a recognized type')


@overload
def pytype_to_abstract(main: tuple, args):
    """Convert a Python type to an AbstractValue."""
    if args == () or args is None:
        targs = ANYTHING
    elif args == ((),):
        targs = []
    else:
        targs = [type_to_abstract(a) for a in args]
    return AbstractTuple(targs)


@overload  # noqa: F811
def pytype_to_abstract(main: list, args):
    arg, = args
    argt = type_to_abstract(arg)
    assert argt is ANYTHING
    rval = AbstractUnion([
        type_to_abstract(Empty),
        type_to_abstract(Cons)
    ])
    return rval


@overload  # noqa: F811
def pytype_to_abstract(main: np.ndarray, args):
    arg, = args
    arg = type_to_abstract(arg)
    shp = ANYTHING
    return AbstractArray(arg, {SHAPE: shp})


@overload  # noqa: F811
def pytype_to_abstract(main: int, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
    })


@overload  # noqa: F811
def pytype_to_abstract(main: float, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Float[64],
    })


@overload  # noqa: F811
def pytype_to_abstract(main: bool, args):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Bool,
    })


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
def abstract_check(self, x: AbstractUnion, *args):
    return self(x.options, *args)


@overload  # noqa: F811
def abstract_check(self, x: Possibilities, *args):
    return all(self(v, *args) for v in x)


@overload  # noqa: F811
def abstract_check(self, x: AbstractTaggedUnion, *args):
    return self(x.options, *args)


@overload  # noqa: F811
def abstract_check(self, x: TaggedPossibilities, *args):
    return all(self(v, *args) for _, v in x)


@overload  # noqa: F811
def abstract_check(self, t: PartialApplication, *args):
    return self(t.fn, *args) and all(self(v, *args) for v in t.args)


@overload  # noqa: F811
def abstract_check(self, t: JTransformedFunction, *args):
    return self(t.fn, *args)


@overload  # noqa: F811
def abstract_check(self, x: Pending, *args):
    return False


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
        if cls is not None:
            inst = cls.empty()
        else:
            inst = None
        constructor = _make_constructor(inst)
        cache[x] = inst
        try:
            result.send(constructor)
        except StopIteration as e:
            if inst is not None:
                assert e.value is inst
            return e.value
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
    elif self.state.check and self.state.check(x, *args):
        return x
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
def abstract_clone(self, x: AbstractDict, *args):
    return (yield AbstractDict)(dict((k, self(v, *args))
                                     for k, v in x.entries.items()))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractArray, *args):
    return (yield type(x))(self(x.element, *args), self(x.values, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractClassBase, *args):
    return (yield type(x))(
        x.tag,
        {k: self(v, *args) for k, v in x.attributes.items()},
        x.methods,
        self(x.values, *args),
        constructor=x.constructor
    )


@overload  # noqa: F811
def abstract_clone(self, x: AbstractUnion, *args):
    return (yield AbstractUnion)(self(x.options, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractTaggedUnion, *args):
    return (yield AbstractTaggedUnion)(self(x.options, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractJTagged, *args):
    return (yield AbstractJTagged)(self(x.element, *args))


@overload  # noqa: F811
def abstract_clone(self, x: AbstractKeywordArgument, *args):
    return (yield AbstractKeywordArgument)(
        x.key,
        self(x.argument, *args)
    )


@overload  # noqa: F811
def abstract_clone(self, x: Possibilities, *args):
    return Possibilities([self(v, *args) for v in x])


@overload  # noqa: F811
def abstract_clone(self, x: TaggedPossibilities, *args):
    return TaggedPossibilities([[i, self(v, *args)] for i, v in x])


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


##############
# Concretize #
##############


@abstract_clone.variant(
    initial_state=lambda: CloneState({}, '_concrete', abstract_check)
)
def concretize_abstract(self, x: Pending):
    """Clone an abstract value while resolving all Pending (synchronous)."""
    if x.done():
        return self(x.result())
    else:
        raise AssertionError('Unresolved Pending', x)


###############
# Broad check #
###############


@abstract_check.variant(
    initial_state=lambda: CheckState(cache={}, prop='_broad')
)
def _is_broad(self, x: object, *args):
    return x is ANYTHING


@overload  # noqa: F811
def _is_broad(self, x: (AbstractScalar, AbstractFunction), *args):
    return self(x.values[VALUE], *args)


###########
# Broaden #
###########


@abstract_clone.variant(
    initial_state=lambda: CloneState({}, '_broad', _is_broad)
)
def broaden(self, d: TrackDict, *args):
    """Broaden an abstract value.

    * Concrete values such as 1 or True will be broadened to ANYTHING.

    Arguments:
        d: The abstract data to clone.
    """
    return {k: k.broaden(v, self, *args) for k, v in d.items()}


###################
# Tentative check #
###################


@_is_broad.variant(
    initial_state=lambda: CheckState(cache={}, prop=None)
)
def _is_tentative(self, x: (Possibilities, TaggedPossibilities), loop):
    return False


#############
# Tentative #
#############


@broaden.variant(
    initial_state=lambda: CloneState({}, None, _is_tentative)
)
def tentative(self, p: Possibilities, loop):
    """Broaden an abstract value and make it tentative.

    * Concrete values such as 1 or True will be broadened to ANYTHING.
    * Possibilities will be broadened to PendingTentative. This allows
      us to avoid resolving them earlier than we would like.

    Arguments:
        d: The abstract data to clone.
        loop: The InferenceLoop, used to broaden Possibilities.
    """
    return loop.create_pending_tentative(tentative=p)


@overload  # noqa: F811
def tentative(self, p: TaggedPossibilities, loop):
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


#################
# Force through #
#################


async def _force_through_post(x):
    return intern(await x)


@overload.wrapper(
    initial_state=lambda: {},
    postprocess=_force_through_post,
)
async def force_through(__call__, self, x, through):
    """Clone an abstract value (asynchronous)."""
    if not isinstance(x, through) and not isinstance(x, Pending):
        return x
    cache = self.state
    if isinstance(x, AbstractValue) and x in cache:
        return cache[x]

    call = __call__(self, x, through)
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


# Uncomment and test the other implementations if/when needed:


# @overload  # noqa: F811
# async def force_through(self, x: AbstractScalar, through):
#     return AbstractScalar(await self(x.values, through))


# @overload  # noqa: F811
# async def force_through(self, x: AbstractFunction, through):
#     yield (yield AbstractFunction)(*(await self(x.get_sync(), through)))


# @overload  # noqa: F811
# async def force_through(self, d: TrackDict, through):
#     return {k: await self(v, through) for k, v in d.items()}


@overload  # noqa: F811
async def force_through(self, x: AbstractTuple, through):
    yield (yield AbstractTuple)(
        [(await self(y, through)) for y in x.elements],
        await self(x.values, through)
    )


@overload  # noqa: F811
async def force_through(self, x: AbstractArray, through):
    yield (yield type(x))(await self(x.element, through),
                          await self(x.values, through))


@overload  # noqa: F811
async def force_through(self, x: AbstractClassBase, through):
    yield (yield type(x))(
        x.tag,
        {k: (await self(v, through)) for k, v in x.attributes.items()},
        x.methods,
        await self(x.values, through)
    )


@overload  # noqa: F811
async def force_through(self, x: AbstractDict, through):
    yield (yield AbstractDict)(
        {k: (await self(v, through)) for k, v in x.entries.items()},
        await self(x.values, through)
    )


@overload  # noqa: F811
async def force_through(self, x: AbstractUnion, through):
    yield (yield AbstractUnion)(await self(x.options, through))


@overload  # noqa: F811
async def force_through(self, x: AbstractTaggedUnion, through):
    opts = await self(x.options, through)
    yield (yield AbstractTaggedUnion)(opts)


@overload  # noqa: F811
async def force_through(self, x: Possibilities, through):
    return Possibilities([await self(v, through) for v in x])


@overload  # noqa: F811
async def force_through(self, x: TaggedPossibilities, through):
    return TaggedPossibilities([[i, await self(v, through)] for i, v in x])


# @overload  # noqa: F811
# async def force_through(self, x: PartialApplication, through):
#     return PartialApplication(
#         await self(x.fn, through),
#         [await self(arg, through) for arg in x.args]
#     )


@overload  # noqa: F811
async def force_through(self, x: Pending, through):
    return await self(await x, through)


############
# Nobottom #
############


@abstract_check.variant
def nobottom(self, x: AbstractBottom):
    """Check whether bottom appears anywhere in this type."""
    return False


@overload  # noqa: F811
def nobottom(self, x: Pending, *args):
    return True


#########
# Merge #
#########


@overload.wrapper(
    bootstrap=True,
    initial_state=set
)
def amerge(__call__, self, x1, x2, forced=False, bind_pending=True,
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
        forced: Whether we are already committed to returning x1 or not.
        bind_pending: Whether we bind two Pending, unresolved values.
        accept_pending: Works the same as bind_pending, but only for the
            top level call.
    """
    if x1 is x2:
        return x1
    keypair = (id(x1), id(x2))
    if keypair in self.state:
        return x1
    else:
        self.state.add(keypair)
    while isinstance(x1, Pending) and x1.done() and not forced:
        x1 = x1.result()
    while isinstance(x2, Pending) and x2.done():
        x2 = x2.result()
    isp1 = isinstance(x1, Pending)
    isp2 = isinstance(x2, Pending)
    loop = x1.get_loop() if isp1 else x2.get_loop() if isp2 else None
    if isinstance(x1, PendingTentative):
        new_tentative = self(x1.tentative, x2, False, True, bind_pending)
        assert not isinstance(new_tentative, Pending)
        x1.tentative = new_tentative
        return x1
    if isinstance(x2, PendingTentative):
        new_tentative = self(x1, x2.tentative, forced,
                             bind_pending, accept_pending)
        assert not isinstance(new_tentative, Pending)
        x2.tentative = new_tentative
        return new_tentative if forced else x2
    if (isp1 or isp2) and (not accept_pending or not bind_pending):
        if forced and isp1:
            raise MyiaTypeError('Cannot have Pending here.')
        if isp1:
            def chk(a):
                return self(a, x2, forced, bind_pending)
            return find_coherent_result_sync(x1, chk)
        if isp2:
            def chk(a):
                return self(x1, a, forced, bind_pending)
            return find_coherent_result_sync(x2, chk)
    if isp1 and isp2:
        return bind(loop, x1 if forced else None, [], [x1, x2])
    elif isp1:
        return bind(loop, x1 if forced else None, [x2], [x1])
    elif isp2:
        return bind(loop, x1 if forced else None, [x1], [x2])
    elif isinstance(x2, AbstractBottom):  # pragma: no cover
        return x1
    elif isinstance(x1, AbstractBottom):
        if forced:  # pragma: no cover
            # I am not sure how to trigger this
            raise TypeMismatchError(x1, x2)
        return x2
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
        return self.map[type(x1)](self, x1, x2, forced, bind_pending)


@overload  # noqa: F811
def amerge(self, x1: Possibilities, x2, forced, bp):
    if set(x1).issuperset(set(x2)):
        return x1
    if forced:
        raise MyiaTypeError(
            'Additional Possibilities cannot be merged.'
        )
    else:
        return Possibilities(x1 + x2)


@overload  # noqa: F811
def amerge(self, x1: TaggedPossibilities, x2, forced, bp):
    d1 = dict(x1)
    d2 = dict(x2)
    results = {}
    for i, t in d1.items():
        if i in d2:
            t = self(t, d2[i], forced, bp)
        results[i] = t
    for i, t in d2.items():
        if i not in d1:
            results[i] = t
    res = TaggedPossibilities(results.items())
    if res == x1:
        return x1
    elif forced:
        raise MyiaTypeError(
            'Additional TaggedPossibilities cannot be merged.'
        )
    elif res == x2:
        return x2
    else:
        return res


@overload  # noqa: F811
def amerge(self, x1: dtype.TypeMeta, x2, forced, bp):
    if issubclass(x2, x1):
        return x1
    elif not forced and issubclass(x1, x2):
        return x2
    else:
        raise TypeMismatchError(x1, x2)


@overload  # noqa: F811
def amerge(self, x1: (dict, TrackDict), x2, forced, bp):
    if set(x1.keys()) != set(x2.keys()):
        # This shouldn't be possible at the moment
        raise AssertionError(f'Keys mismatch')
    changes = False
    rval = type(x1)()
    for k, v in x1.items():
        res = self(v, x2[k], forced, bp)
        if res is not v:
            changes = True
        rval[k] = res
    return x1 if forced or not changes else rval


@overload  # noqa: F811
def amerge(self, x1: (tuple, list), x2, forced, bp):
    if len(x1) != len(x2):  # pragma: no cover
        raise MyiaTypeError(f'Tuple length mismatch')
    changes = False
    rval = []
    for v1, v2 in zip(x1, x2):
        res = self(v1, v2, forced, bp)
        if res is not v1:
            changes = True
        rval.append(res)
    return x1 if forced or not changes else type(x1)(rval)


@overload  # noqa: F811
def amerge(self, x1: AbstractScalar, x2, forced, bp):
    values = self(x1.values, x2.values, forced, bp)
    if forced or values is x1.values:
        return x1
    return AbstractScalar(values)


@overload  # noqa: F811
def amerge(self, x1: AbstractError, x2, forced, bp):
    e1 = x1.values[VALUE]
    e2 = x2.values[VALUE]
    e = self(e1, e2, forced, bp)
    if forced or e is e1:
        return x1
    return AbstractError(e)


@overload  # noqa: F811
def amerge(self, x1: AbstractFunction, x2, forced, bp):
    values = self(x1.get_sync(), x2.get_sync(), forced, bp)
    if forced or values is x1.values:
        return x1
    return AbstractFunction(*values)


@overload  # noqa: F811
def amerge(self, x1: AbstractTuple, x2, forced, bp):
    args1 = (x1.elements, x1.values)
    args2 = (x2.elements, x2.values)
    merged = self(args1, args2, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractTuple(*merged)


@overload  # noqa: F811
def amerge(self, x1: AbstractArray, x2, forced, bp):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = self(args1, args2, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractArray(*merged)


@overload  # noqa: F811
def amerge(self, x1: AbstractClassBase, x2, forced, bp):
    args1 = (x1.tag, x1.attributes, x1.methods, x1.values)
    args2 = (x2.tag, x2.attributes, x2.methods, x2.values)
    merged = self(args1, args2, forced, bp)
    if forced or merged is args1:
        return x1
    return type(x1)(*merged)


@overload  # noqa: F811
def amerge(self, x1: AbstractJTagged, x2, forced, bp):
    args1 = x1.element
    args2 = x2.element
    merged = self(args1, args2, forced, bp)
    if forced or merged is args1:
        return x1
    return AbstractJTagged(merged)


@overload  # noqa: F811
def amerge(self, x1: (AbstractUnion, AbstractTaggedUnion),
           x2, forced, bp):
    args1 = x1.options
    args2 = x2.options
    merged = self(args1, args2, forced, bp)
    if forced or merged is args1:
        return x1
    return type(x1)(merged)


@overload  # noqa: F811
def amerge(self, x1: (int, float, bool), x2, forced, bp):
    if forced and x1 != x2:
        raise TypeMismatchError(x1, x2)
    return x1 if x1 == x2 else ANYTHING


@overload  # noqa: F811
def amerge(self, x1: object, x2, forced, bp):
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
                                             forced=False,
                                             accept_pending=False),
                       resolved)
        else:
            v = reduce(lambda x1, x2: amerge(x1, x2,
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
        committed = tentative(committed, loop)
        resolved.clear()
        return committed

    def priority():
        # Cannot force resolve unless we have at least one resolved Pending
        if not resolved and committed is None:
            return None
        if any(is_simple(x) for x in chain([committed], resolved, pending)):
            return 1000
        elif any(not nobottom(x) for x in resolved):
            # Bottom is always lower-priority
            return None
        else:
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


def collapse_options(options):
    """Collapse a list of options, some of which may be AbstractUnions."""
    opts = []
    todo = list(options)
    while todo:
        option = todo.pop()
        if isinstance(option, AbstractUnion):
            todo.extend(option.options)
        else:
            opts.append(option)
    opts = Possibilities(opts)
    return opts


def union_simplify(options, constructor=AbstractUnion):
    """Simplify a list of options.

    Returns:
        * None, if there are no options.
        * A single type, if there is one option.
        * An AbstractUnion.

    """
    options = collapse_options(options)
    if len(options) == 0:
        return None
    elif len(options) == 1:
        return options.pop()
    else:
        return constructor(options)


def typecheck(model, abstract):
    """Check that abstract matches the model."""
    try:
        amerge(model, abstract, forced=True, bind_pending=False)
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
        t1 = union_simplify(opt for opt, m in matching if m)
        t2 = union_simplify(opt for opt, m in matching if not m)
        return t1, t2
    elif typecheck(model, t):
        return t, None
    else:
        return None, t


def hastype_helper(value, model):
    """Helper to implement hastype."""
    if isinstance(model, AbstractUnion):
        results = [hastype_helper(value, opt) for opt in model.options]
        if any(r is True for r in results):
            return True
        elif all(r is False for r in results):
            return False
        else:
            return ANYTHING
    else:
        match, nomatch = split_type(value, model)
        if match is None:
            return False
        elif nomatch is None:
            return True
        else:
            return ANYTHING


#########################
# ADT-related utilities #
#########################


def normalize_adt(x):
    """Normalize the ADT to make it properly recursive."""
    rval = _normalize_adt_helper(x, {}, {})
    rval = rval.intern()
    rval = broaden(rval)
    rval = _finalize_adt(rval)
    return rval


def _normalize_adt_helper(x, done, tag_to_adt):
    if x in done:
        return done[x]
    if isinstance(x, AbstractADT):
        if x.tag not in tag_to_adt:
            adt = AbstractADT.new(
                x.tag,
                {k: AbstractUnion.new([]) for k in x.attributes},
                x.methods
            )
            tag_to_adt = {**tag_to_adt, x.tag: adt}
        else:
            adt = tag_to_adt[x.tag]
        done[x] = adt
        for attr, value in x.attributes.items():
            value = _normalize_adt_helper(value, done, tag_to_adt)
            adt.attributes[attr] = union_simplify(
                [adt.attributes[attr], value],
                constructor=AbstractUnion.new
            )
        return adt
    elif isinstance(x, AbstractUnion):
        opts = _normalize_adt_helper(x.options, done, tag_to_adt)
        rval = union_simplify(opts, constructor=AbstractUnion.new)
        done[x] = rval
        return rval
    elif isinstance(x, Possibilities):
        return [_normalize_adt_helper(opt, done, tag_to_adt) for opt in x]
    else:
        return x


@abstract_clone.variant
def _finalize_adt(self, x: AbstractUnion):
    x = union_simplify(x.options)
    if isinstance(x, AbstractUnion):
        return (yield AbstractUnion)(self(x.options))
    else:
        yield None
        return self(x)
