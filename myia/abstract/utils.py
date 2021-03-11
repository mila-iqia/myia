from collections import defaultdict

from ovld import ovld

from . import data
from .map import MapError, abstract_all, abstract_map, abstract_map2

###################
# no_placeholders #
###################


@abstract_all.variant(
    initial_state=lambda: {"cache": {}, "prop": None},
)
def no_placeholders(self, ph: data.Placeholder, **_):
    return False


#############
# canonical #
#############


@abstract_map.variant(
    initial_state=lambda: {
        "cache": {},
        "remap": {},
        "prop": None,
        "check": no_placeholders,
    }
)
def canonical(self, ph: data.Placeholder):
    if ph not in self.remap:
        self.remap[ph] = data.Opaque(len(self.remap))
    return self.remap[ph]


###############
# uncanonical #
###############


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": None, "check": None}
)
def uncanonical(self, opq: data.Opaque):
    # Only called once per call to uncanonical per distinct opq, because the
    # result is cached in self.cache. Each call to uncanonical has its own
    # cache, however.
    return data.Placeholder()


#########
# reify #
#########


@abstract_all.variant
def is_concrete(self, _: (data.Opaque, data.Placeholder), **kwargs):
    return False


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": None, "check": is_concrete}
)
def reify(self, opq: data.Opaque, *, unif):
    sub = unif.get(opq, opq)
    if sub is opq:
        return sub
    else:
        return self(sub, unif=unif)


#########
# unify #
#########


class Unificator:
    def __init__(self):
        self.eqv = defaultdict(set)
        self.canon = {}

    def unify(self, ufn, x, y):
        if y in self.eqv[x]:
            return self.canon[x]

        eqv = self.eqv[x] | self.eqv[y] | {x, y}
        self.eqv[x] = self.eqv[y] = eqv

        cx = self.canon.get(x, x)
        cy = self.canon.get(y, y)

        if isinstance(cx, data.Generic):
            merged = cy
        elif isinstance(cy, data.Generic):
            merged = cx
        else:
            merged = ufn(cx, cy, U=self)

        eqv.add(merged)
        for entry in eqv:
            self.canon[entry] = merged

        return merged


@abstract_map2.variant
def _unify(self, x: data.Generic, y: object, *, U):
    return U.unify(self, x, y)


@ovld
def _unify(self, x: object, y: data.Generic, *, U):
    return U.unify(self, x, y)


@ovld
def _unify(self, x: data.Generic, y: data.Generic, *, U):
    return U.unify(self, x, y)


@ovld
def _unify(self, x: object, y: object, *, U):
    if x == y:
        return x
    else:
        raise MapError(x, y, reason="Cannot merge objects")


def unify(x, y, U=None):
    U = U or Unificator()
    res = _unify(x, y, U=U)
    return reify(res, unif=U.canon), U


#########
# merge #
#########


@_unify.variant
def _merge(self, x: data.AbstractUnion, y: data.AbstractUnion, *, U):
    assert type(x) is type(y)
    return (yield type(x))(
        [*x.options, *y.options], tracks=self(x.tracks, y.tracks, U=U)
    )


def merge(x, y, U=None):
    U = U or Unificator()
    res = _merge(x, y, U=U)
    return reify(res, unif=U.canon), U
