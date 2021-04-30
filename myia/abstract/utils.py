"""Utilities on abstract data."""

from collections import defaultdict

from ovld import ovld

from . import data
from .map import MapError, abstract_all, abstract_map, abstract_map2

#############
# canonical #
#############


@abstract_map.variant
def canonical(self, gn: data.Generic, *, mapping):
    """Return an AbstractValue with canonical generics.

    Essentially remaps the first encountered generic to Opaque(0), the
    second to Opaque(1), etc.

    The mapping from generic to canonical is stored in mapping.
    """
    if gn not in mapping:
        mapping[gn] = data.Opaque(len(mapping))
    return mapping[gn]


###############
# uncanonical #
###############


@abstract_map.variant
def _uncanonical(self, gn: data.Generic, *, invmapping):
    """Undo the canonical transform."""
    return invmapping[gn]


def uncanonical(x, *, mapping):
    """Undo the mapping of canonical.

    uncanonical(canonical(x, mapping), mapping) is x.
    """
    invmapping = {v: k for k, v in mapping.items()}
    return _uncanonical(x, invmapping=invmapping)


##################
# fresh_generics #
##################


@abstract_map.variant
def fresh_generics(self, gn: data.Generic, *, mapping):
    """Map every generic to a fresh Placeholder."""
    if gn not in mapping:
        mapping[gn] = data.Placeholder()
    return mapping[gn]


#########
# reify #
#########


@abstract_all.variant
def is_concrete(self, _: data.Generic, **kwargs):
    """Return whether an AbstractValue contains any generics."""
    return False


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": None, "check": is_concrete}
)
def reify(self, gn: data.Generic, *, unif):
    """Replace generics by the concrete type they correspond to."""
    sub = unif.get(gn, gn)
    if sub is gn:
        return sub
    else:
        return self(sub, unif=unif)


#########
# unify #
#########


class Unificator:
    """Class to perform unification.

    Attributes:
        eqv: Dictionary of equivalence classes.
        canon: Map variables to a canonical variable or result.
    """

    def __init__(self):
        self.eqv = defaultdict(set)
        self.canon = {}

    def unify(self, ufn, x, y):
        """Unify x and y, using the unification function."""
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
def _unify(self, x: object, y: data.Generic, *, U):  # noqa: F811
    return U.unify(self, x, y)


@ovld
def _unify(self, x: data.Generic, y: data.Generic, *, U):  # noqa: F811
    return U.unify(self, x, y)


@ovld
def _unify(self, x: object, y: object, *, U):  # noqa: F811
    if x == y:
        return x
    else:
        raise MapError(x, y, reason="Cannot merge objects")


def unify(x, y, U=None):
    """Unify x with y using Unificator U."""
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
    """Merge x with y using Unificator U.

    Works like unify except for AbstractUnion: merge(union1, union2) will
    append both unions' possibilities rather than unify them.
    """
    U = U or Unificator()
    res = _merge(x, y, U=U)
    return reify(res, unif=U.canon), U
