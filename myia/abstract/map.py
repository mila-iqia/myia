"""General functions to process or transform AbstractValues."""

from types import GeneratorType

from ovld import ovld

from ..utils.intern import intern
from . import data

#############
# Predicate #
#############


@ovld.dispatch(initial_state=lambda: {"cache": {}, "prop": None})
def abstract_predicate(self, x, **kwargs):
    """Check that a predicate applies to a given object."""
    __call__ = self.resolve(x)

    def proceed():
        if prop:
            if hasattr(x, prop):
                return getattr(x, prop) is x
            elif __call__(x, **kwargs):
                if isinstance(x, data.Cachable):
                    setattr(x, prop, x)
                return True
            else:
                return False
        else:
            return __call__(x, **kwargs)

    prop = self.prop
    cache = self.cache

    try:
        rval = cache.get(x, None)
    except TypeError:  # pragma: no cover
        return proceed()

    if rval is None:
        cache[x] = True
        cache[x] = proceed()
        return cache[x]
    else:
        return rval


#######
# All #
#######


@abstract_predicate.variant
def abstract_all(self, x: data.Tracks, **kwargs):
    """Check that a predicate applies recursively to the whole structure."""
    return all(self(v, **kwargs) for v in x.values())


@ovld
def abstract_all(self, x: data.AbstractAtom, **kwargs):  # noqa: F811
    return self(x.tracks, **kwargs)


@ovld
def abstract_all(self, xs: data.AbstractStructure, **kwargs):  # noqa: F811
    return self(xs.tracks, **kwargs) and all(
        self(x, **kwargs) for x in xs.elements
    )


@ovld
def abstract_all(self, xs: data.AbstractUnion, **kwargs):  # noqa: F811
    return self(xs.tracks, **kwargs) and all(
        self(x, **kwargs) for x in xs.options
    )


@ovld
def abstract_all(self, xs: object, **kwargs):  # noqa: F811
    return True


#######
# Any #
#######


@abstract_predicate.variant
def abstract_any(self, x: data.Tracks, **kwargs):
    """Check that a predicate applies to part of the structure."""
    return any(self(v, **kwargs) for v in x.values())


@ovld
def abstract_any(self, x: data.AbstractAtom, **kwargs):  # noqa: F811
    return self(x.tracks, **kwargs)


@ovld
def abstract_any(self, xs: data.AbstractStructure, **kwargs):  # noqa: F811
    return self(xs.tracks, **kwargs) or any(
        self(x, **kwargs) for x in xs.elements
    )


@ovld
def abstract_any(self, xs: data.AbstractUnion, **kwargs):  # noqa: F811
    return self(xs.tracks, **kwargs) or any(
        self(x, **kwargs) for x in xs.options
    )


@ovld
def abstract_any(self, xs: object, **kwargs):  # noqa: F811
    return False


#######
# Map #
#######


def _make_constructor(inst):
    def f(*args, **kwargs):
        inst.commit(*args, **kwargs)
        return inst

    return f


def _intern(_, x):
    return intern(x)


@ovld.dispatch(
    initial_state=lambda: {"cache": {}, "prop": None, "check": None},
    postprocess=_intern,
)
def abstract_map(self, x, **kwargs):
    """Clone an abstract value."""
    __call__ = self.resolve(x)

    def proceed():
        if isinstance(x, data.Cachable) and x in cache:
            return cache[x]
        result = __call__(x, **kwargs)
        if not isinstance(result, GeneratorType):
            if isinstance(x, data.Cachable):
                cache[x] = result
            return result
        cls = result.send(None)
        assert cls is not None
        inst = cls.empty()
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
                "Generators in abstract_map must yield once, then return."
            )

    cache = self.cache
    prop = self.prop
    if prop:
        if hasattr(x, prop):
            return getattr(x, prop)
        elif isinstance(x, data.AbstractValue):
            if self.check(x, **kwargs):
                res = x
            else:
                res = proceed()
            setattr(x, prop, res)
            return res
        else:
            return proceed()
    elif self.check and self.check(x, **kwargs):
        return x
    else:
        return proceed()


@ovld
def abstract_map(self, d: data.Tracks, **kwargs):  # noqa: F811
    return data.Tracks({k: self(v, **kwargs) for k, v in d.items()})


@ovld
def abstract_map(self, x: data.AbstractAtom, **kwargs):  # noqa: F811
    return type(x)(self(x.tracks, **kwargs))


@ovld
def abstract_map(self, x: data.AbstractStructure, **kwargs):  # noqa: F811
    return (yield type(x))(
        [self(elem, **kwargs) for elem in x.elements], self(x.tracks, **kwargs)
    )


@ovld
def abstract_map(self, x: data.AbstractUnion, **kwargs):  # noqa: F811
    return (yield type(x))(
        [self(opt, **kwargs) for opt in x.options], self(x.tracks, **kwargs)
    )


@ovld
def abstract_map(self, x: object, **kwargs):  # noqa: F811
    return x


########
# Map2 #
########


class MapError(Exception):
    """Represents a matching error between two objects."""

    def __init__(self, x, y, reason):
        super().__init__(reason)
        self.x = x
        self.y = y
        self.reason = reason


@ovld.dispatch(
    initial_state=lambda: {"cache": {}},
    postprocess=_intern,
)
def abstract_map2(self, x, y, **kwargs):
    """Combine two abstract values."""
    __call__ = self.resolve(x, y)

    cache = self.cache
    cachable = isinstance(x, data.Cachable) and isinstance(y, data.Cachable)

    if cachable and (x, y) in cache:
        return cache[x, y]

    result = __call__(x, y, **kwargs)
    if not isinstance(result, GeneratorType):
        if cachable:
            cache[x, y] = result
        return result

    cls = result.send(None)
    assert cls is not None
    inst = cls.empty()
    constructor = _make_constructor(inst)
    cache[x, y] = inst

    try:
        result.send(constructor)
    except StopIteration as e:
        if inst is not None:
            assert e.value is inst
        return e.value
    else:
        raise AssertionError(
            "Generators in abstract_map2 must yield once, then return."
        )


@ovld
def abstract_map2(self, x: data.Tracks, y: data.Tracks, **kwargs):  # noqa: F811
    tracks = {**x._tracks, **y._tracks}
    return data.Tracks(
        {
            k: self(x.get_track(k), y.get_track(k), **kwargs)
            for k in tracks.keys()
        }
    )


@ovld
def abstract_map2(  # noqa: F811
    self, x: data.AbstractAtom, y: data.AbstractAtom, **kwargs
):
    assert type(x) is type(y)
    return type(x)(tracks=self(x.tracks, y.tracks, **kwargs))


@ovld
def abstract_map2(  # noqa: F811
    self, x: data.AbstractStructure, y: data.AbstractStructure, **kwargs
):
    assert type(x) is type(y)
    if len(x.elements) != len(y.elements):
        raise MapError(x, y, reason="Structures have different lengths")

    return (yield type(x))(
        [self(xe, ye, **kwargs) for xe, ye in zip(x.elements, y.elements)],
        tracks=self(x.tracks, y.tracks, **kwargs),
    )


@ovld
def abstract_map2(  # noqa: F811
    self, x: data.AbstractUnion, y: data.AbstractUnion, **kwargs
):
    # TODO: this should be more like merging sets

    assert type(x) is type(y)
    if len(x.options) != len(y.options):
        raise MapError(x, y, reason="Unions have different lengths")

    return (yield type(x))(
        [self(xe, ye, **kwargs) for xe, ye in zip(x.options, y.options)],
        tracks=self(x.tracks, y.tracks, **kwargs),
    )


@ovld
def abstract_map2(self, x: object, y: object, **kwargs):  # noqa: F811
    raise MapError(x, y, reason="Cannot merge objects")
