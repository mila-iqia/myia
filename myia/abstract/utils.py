"""Utilities for abstract values and inference."""

from dataclasses import dataclass, replace as dc_replace
from types import AsyncGeneratorType, GeneratorType

from ovld import ovld

from .. import xtype
from ..utils import intern
from .data import (
    ABSENT,
    ANYTHING,
    TYPE,
    VALUE,
    AbstractADT,
    AbstractArray,
    AbstractAtom,
    AbstractClass,
    AbstractClassBase,
    AbstractDict,
    AbstractFunction,
    AbstractFunctionUnique,
    AbstractJTagged,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractStructure,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractType,
    AbstractUnion,
    AbstractValue,
    AbstractWrapper,
    GraphFunction,
    PartialApplication,
    Possibilities,
    TaggedPossibilities,
    TrackDict,
    TransformedFunction,
)
from .loop import Pending
from .ref import Context, Reference

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

    if isinstance(a, AbstractType):
        return a.element

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


@ovld
def _build_value(x: AbstractValue):
    raise ValueError(x)


@ovld  # noqa: F811
def _build_value(x: AbstractTuple):
    return tuple(build_value(y) for y in x.elements)


@ovld  # noqa: F811
def _build_value(ac: AbstractClass):
    args = {k: build_value(v) for k, v in ac.attributes.items()}
    return ac.constructor(**args)


############
# Checking #
############


@dataclass
class CheckState:
    """State of abstract_check."""

    cache: dict
    prop: str


@ovld.dispatch(initial_state=lambda: {"state": CheckState({}, None)})
def abstract_check(self, x, **kwargs):
    """Check that a predicate applies to a given object."""
    __call__ = self.resolve(x)

    def proceed():
        if prop:
            if hasattr(x, prop):
                return getattr(x, prop) is x
            elif __call__(x, **kwargs):
                if isinstance(x, AbstractValue):
                    setattr(x, prop, x)
                return True
            else:
                return False
        else:
            return __call__(x, **kwargs)

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


@ovld  # noqa: F811
def abstract_check(self, x: TrackDict, **kwargs):
    return all(self(v, **kwargs) for v in x.values())


@ovld  # noqa: F811
def abstract_check(self, x: AbstractScalar, **kwargs):
    return self(x.values, **kwargs)


@ovld  # noqa: F811
def abstract_check(self, xs: AbstractStructure, **kwargs):
    ch = xs.children()
    if ch is ANYTHING:
        return self(ch)
    else:
        return all(self(x, **kwargs) for x in ch)


@ovld  # noqa: F811
def abstract_check(self, xs: AbstractAtom, **kwargs):
    return True


@ovld  # noqa: F811
def abstract_check(self, x: AbstractFunction, **kwargs):
    return self(x.values, **kwargs)


@ovld  # noqa: F811
def abstract_check(self, x: AbstractFunctionUnique, **kwargs):
    return (
        self(x.values, **kwargs)
        and all(self(v, **kwargs) for v in x.args)
        and self(x.output, **kwargs)
    )


@ovld  # noqa: F811
def abstract_check(self, x: AbstractUnion, **kwargs):
    return self(x.options, **kwargs)


@ovld  # noqa: F811
def abstract_check(self, x: Possibilities, **kwargs):
    return all(self(v, **kwargs) for v in x)


@ovld  # noqa: F811
def abstract_check(self, x: AbstractTaggedUnion, **kwargs):
    return self(x.options, **kwargs)


@ovld  # noqa: F811
def abstract_check(self, x: TaggedPossibilities, **kwargs):
    return all(self(v, **kwargs) for _, v in x)


@ovld  # noqa: F811
def abstract_check(self, t: PartialApplication, **kwargs):
    return self(t.fn, **kwargs) and all(self(v, **kwargs) for v in t.args)


@ovld  # noqa: F811
def abstract_check(self, t: TransformedFunction, **kwargs):
    return self(t.fn, **kwargs)


@ovld  # noqa: F811
def abstract_check(self, x: Pending, **kwargs):
    return False


@ovld  # noqa: F811
def abstract_check(self, xs: object, **kwargs):
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


def _intern(_, x):
    return intern(x)


@ovld.dispatch(
    initial_state=lambda: {"state": CloneState({}, None, None)},
    postprocess=_intern,
)
def abstract_clone(self, x, **kwargs):
    """Clone an abstract value."""
    __call__ = self.resolve(x)

    def proceed():
        if isinstance(x, AbstractValue) and x in cache:
            return cache[x]
        result = __call__(x, **kwargs)
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
                "Generators in abstract_clone must yield once, then return."
            )

    cache = self.state.cache
    prop = self.state.prop
    if prop:
        if hasattr(x, prop):
            return getattr(x, prop)
        elif isinstance(x, AbstractValue):
            if self.state.check(x, **kwargs):
                res = x
            else:
                res = proceed()
            setattr(x, prop, res)
            return res
        else:
            return proceed()
    elif self.state.check and self.state.check(x, **kwargs):
        return x
    else:
        return proceed()


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractScalar, **kwargs):
    return AbstractScalar(self(x.values, **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractFunction, **kwargs):
    return (yield AbstractFunction)(value=self(x.get_sync(), **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, d: TrackDict, **kwargs):
    return {k: k.clone(v, self) for k, v in d.items()}


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractTuple, **kwargs):
    if x.elements is ANYTHING:
        return (yield AbstractTuple)(ANYTHING)
    return (yield AbstractTuple)(
        [self(y, **kwargs) for y in x.elements], self(x.values, **kwargs)
    )


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractDict, **kwargs):
    return (yield AbstractDict)(
        dict((k, self(v, **kwargs)) for k, v in x.entries.items())
    )


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractWrapper, **kwargs):
    return (yield type(x))(self(x.element, **kwargs), self(x.values, **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractClassBase, **kwargs):
    return (yield type(x))(
        x.tag,
        {k: self(v, **kwargs) for k, v in x.attributes.items()},
        values=self(x.values, **kwargs),
        constructor=x.constructor,
    )


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractUnion, **kwargs):
    return (yield AbstractUnion)(self(x.options, **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractTaggedUnion, **kwargs):
    return (yield AbstractTaggedUnion)(self(x.options, **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractKeywordArgument, **kwargs):
    return (yield AbstractKeywordArgument)(x.key, self(x.argument, **kwargs))


@ovld  # noqa: F811
def abstract_clone(self, x: Possibilities, **kwargs):
    return Possibilities([self(v, **kwargs) for v in x])


@ovld  # noqa: F811
def abstract_clone(self, x: TaggedPossibilities, **kwargs):
    return TaggedPossibilities([[i, self(v, **kwargs)] for i, v in x])


@ovld  # noqa: F811
def abstract_clone(self, x: PartialApplication, **kwargs):
    return PartialApplication(
        self(x.fn, **kwargs), [self(arg, **kwargs) for arg in x.args]
    )


@ovld  # noqa: F811
def abstract_clone(self, x: TransformedFunction, **kwargs):
    return TransformedFunction(self(x.fn, **kwargs), x.transform)


@ovld  # noqa: F811
def abstract_clone(self, x: AbstractFunctionUnique, **kwargs):
    return (yield AbstractFunctionUnique)(
        [self(arg, **kwargs) for arg in x.args], self(x.output, **kwargs)
    )


@ovld  # noqa: F811
def abstract_clone(self, x: Pending, **kwargs):
    if x.done():
        return self(x.result(), **kwargs)
    else:
        return x


@ovld  # noqa: F811
def abstract_clone(self, x: object, **kwargs):
    return x


##############
# Concretize #
##############


@abstract_clone.variant(
    initial_state=lambda: {"state": CloneState({}, "_concrete", abstract_check)}
)
def concretize_abstract(self, x: Pending):
    """Clone an abstract value while resolving all Pending (synchronous)."""
    if x.done():
        return self(x.result())
    else:
        raise AssertionError("Unresolved Pending", x)


@abstract_check.variant(
    initial_state=lambda: {"state": CheckState({}, "_no_track")}
)
def _check_no_tracking_id(self, x: GraphFunction):
    return x.tracking_id is None


@concretize_abstract.variant(
    initial_state=lambda: {
        "state": CloneState(
            cache={}, prop="_no_track", check=_check_no_tracking_id
        )
    }
)
def no_tracking_id(self, x: GraphFunction):
    """Resolve all Pending and erase tracking_id information."""
    return dc_replace(x, tracking_id=None)


def concretize_cache(src, dest=None):
    """Complete a cache with concretized versions of its keys.

    If an entry in the cache has a key that contains a Pending, a new key
    is created where the Pending is resolved, and it is entered in the cache
    so that it can be found more easily.
    """
    if dest is None:
        dest = src
    for k, v in list(src.items()):
        kc = refmap(concretize_abstract, k)
        dest[kc] = v
        kc2 = refmap(no_tracking_id, kc)
        dest[kc2] = v


###############
# Broad check #
###############


@abstract_check.variant(
    initial_state=lambda: {"state": CheckState(cache={}, prop="_broad")}
)
def is_broad(self, x: object, **kwargs):
    """Check whether the object is broad or not."""
    return x is ANYTHING


@ovld  # noqa: F811
def is_broad(self, x: (AbstractScalar, AbstractFunction), **kwargs):
    return self(x.xvalue(), **kwargs)


###########
# Broaden #
###########


@abstract_clone.variant(
    initial_state=lambda: {"state": CloneState({}, "_broad", is_broad)}
)
def broaden(self, d: TrackDict, **kwargs):  # noqa: D417
    """Broaden an abstract value.

    * Concrete values such as 1 or True will be broadened to ANYTHING.

    Arguments:
        d: The abstract data to clone.

    """
    return {k: k.broaden(v, self, **kwargs) for k, v in d.items()}


###############
# Sensitivity #
###############


@abstract_clone.variant
def sensitivity_transform(self, x: (AbstractFunction, AbstractFunctionUnique)):
    """Return an abstract value for the sensitivity of x.

    * The sensitivity of a function is an Env
    * The sensitivity of J(x) is x
    """
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.EnvType})


@ovld  # noqa: F811
def sensitivity_transform(self, x: AbstractJTagged):
    return self(x.element)


#################
# Force through #
#################


async def _force_through_post(_, x):
    return intern(await x)


@ovld.dispatch(
    initial_state=lambda: {"state": {}}, postprocess=_force_through_post
)
async def force_through(self, x, through):
    """Clone an abstract value (asynchronous)."""
    __call__ = self[type(x), object]
    if not isinstance(x, through) and not isinstance(x, Pending):
        return x
    cache = self.state
    if isinstance(x, AbstractValue) and x in cache:
        return cache[x]

    call = __call__(x, through)
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


# @ovld  # noqa: F811
# async def force_through(self, x: AbstractScalar, through):
#     return AbstractScalar(await self(x.values, through))


# @ovld  # noqa: F811
# async def force_through(self, x: AbstractFunction, through):
#     yield (yield AbstractFunction)(*(await self(x.get_sync(), through)))


# @ovld  # noqa: F811
# async def force_through(self, d: TrackDict, through):
#     return {k: await self(v, through) for k, v in d.items()}


@ovld  # noqa: F811
async def force_through(self, x: AbstractTuple, through):
    yield (yield AbstractTuple)(
        [(await self(y, through)) for y in x.elements],
        await self(x.values, through),
    )


@ovld  # noqa: F811
async def force_through(self, x: AbstractArray, through):
    yield (yield type(x))(
        await self(x.element, through), await self(x.values, through)
    )


@ovld  # noqa: F811
async def force_through(self, x: AbstractClassBase, through):
    yield (yield type(x))(
        x.tag,
        {k: (await self(v, through)) for k, v in x.attributes.items()},
        values=await self(x.values, through),
    )


@ovld  # noqa: F811
async def force_through(self, x: AbstractDict, through):
    yield (yield AbstractDict)(
        {k: (await self(v, through)) for k, v in x.entries.items()},
        await self(x.values, through),
    )


@ovld  # noqa: F811
async def force_through(self, x: AbstractUnion, through):
    yield (yield AbstractUnion)(await self(x.options, through))


@ovld  # noqa: F811
async def force_through(self, x: AbstractTaggedUnion, through):
    opts = await self(x.options, through)
    yield (yield AbstractTaggedUnion)(opts)


@ovld  # noqa: F811
async def force_through(self, x: Possibilities, through):
    return Possibilities([await self(v, through) for v in x])


@ovld  # noqa: F811
async def force_through(self, x: TaggedPossibilities, through):
    return TaggedPossibilities([[i, await self(v, through)] for i, v in x])


# @ovld  # noqa: F811
# async def force_through(self, x: PartialApplication, through):
#     return PartialApplication(
#         await self(x.fn, through),
#         [await self(arg, through) for arg in x.args]
#     )


@ovld  # noqa: F811
async def force_through(self, x: Pending, through):
    return await self(await x, through)


@ovld  # noqa: F811
async def force_through(self, x: object, through):
    raise NotImplementedError(type(x))


################################
# Map a function on references #
################################


@ovld
def refmap(self, fn, x: Context):
    """Map a function on a Reference/Context/etc."""
    return Context(
        self(fn, x.parent), x.graph, tuple(fn(arg) for arg in x.argkey)
    )


@ovld  # noqa: F811
def refmap(self, fn, x: Reference):
    return Reference(x.engine, x.node, self(fn, x.context))


@ovld  # noqa: F811
def refmap(self, fn, x: tuple):
    return tuple(self(fn, y) for y in x)


@ovld  # noqa: F811
def refmap(self, fn, x: AbstractValue):
    return fn(x)


@ovld  # noqa: F811
def refmap(self, fn, x: object):
    return x


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
                x.tag, {k: AbstractUnion.new([]) for k in x.attributes}
            )
            tag_to_adt = {**tag_to_adt, x.tag: adt}
        else:
            adt = tag_to_adt[x.tag]
        done[x] = adt
        for attr, value in x.attributes.items():
            value = _normalize_adt_helper(value, done, tag_to_adt)
            adt.attributes[attr] = union_simplify(
                [adt.attributes[attr], value], constructor=AbstractUnion.new
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


__consolidate__ = True
__all__ = [
    "CheckState",
    "CloneState",
    "abstract_check",
    "abstract_clone",
    "broaden",
    "build_value",
    "collapse_options",
    "concretize_abstract",
    "concretize_cache",
    "force_through",
    "is_broad",
    "no_tracking_id",
    "normalize_adt",
    "refmap",
    "sensitivity_transform",
    "union_simplify",
]
