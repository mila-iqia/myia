from types import GeneratorType

from ovld import ovld

from ..utils.intern import intern
from . import data

#########
# Check #
#########


@ovld.dispatch(initial_state=lambda: {"cache": {}, "prop": None})
def abstract_predicate(self, x, **kwargs):
    """Check that a predicate applies to a given object."""
    __call__ = self.resolve(x)

    def proceed():
        if prop:
            if hasattr(x, prop):
                return getattr(x, prop) is x
            elif __call__(x, **kwargs):
                if isinstance(x, data.AbstractValue):
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
    return all(self(v, **kwargs) for v in x.values())


@ovld  # noqa: F811
def abstract_all(self, x: data.AbstractAtom, **kwargs):
    return self(x.tracks, **kwargs)


@ovld  # noqa: F811
def abstract_all(self, xs: data.AbstractStructure, **kwargs):
    return self(xs.tracks, **kwargs) and all(
        self(x, **kwargs) for x in xs.elements
    )


@ovld  # noqa: F811
def abstract_all(self, xs: data.AbstractUnion, **kwargs):
    return self(xs.tracks, **kwargs) and all(
        self(x, **kwargs) for x in xs.options
    )


@ovld  # noqa: F811
def abstract_all(self, xs: object, **kwargs):
    return True


#######
# Any #
#######


@abstract_predicate.variant
def abstract_any(self, x: data.Tracks, **kwargs):
    return any(self(v, **kwargs) for v in x.values())


@ovld  # noqa: F811
def abstract_any(self, x: data.AbstractAtom, **kwargs):
    return self(x.tracks, **kwargs)


@ovld  # noqa: F811
def abstract_any(self, xs: data.AbstractStructure, **kwargs):
    return self(xs.tracks, **kwargs) or any(
        self(x, **kwargs) for x in xs.elements
    )


@ovld  # noqa: F811
def abstract_any(self, xs: data.AbstractUnion, **kwargs):
    return self(xs.tracks, **kwargs) or any(
        self(x, **kwargs) for x in xs.options
    )


@ovld  # noqa: F811
def abstract_any(self, xs: object, **kwargs):
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
        if isinstance(x, data.AbstractValue) and x in cache:
            return cache[x]
        result = __call__(x, **kwargs)
        if not isinstance(result, GeneratorType):
            if isinstance(x, data.AbstractValue):
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


@ovld  # noqa: F811
def abstract_map(self, d: data.Tracks, **kwargs):
    return data.Tracks({k: self(v, **kwargs) for k, v in d.items()})


@ovld  # noqa: F811
def abstract_map(self, x: data.AbstractAtom, **kwargs):
    return type(x)(self(x.tracks, **kwargs))


@ovld  # noqa: F811
def abstract_map(self, x: data.AbstractStructure, **kwargs):
    return (yield type(x))(
        [self(elem, **kwargs) for elem in x.elements], self(x.tracks, **kwargs)
    )


@ovld  # noqa: F811
def abstract_map(self, x: data.AbstractUnion, **kwargs):
    return (yield type(x))(
        [self(opt, **kwargs) for opt in x.options], self(x.tracks, **kwargs)
    )


@ovld  # noqa: F811
def abstract_map(self, x: object, **kwargs):
    return x
