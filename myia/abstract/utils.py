"""Utilities on abstract data."""

from collections import defaultdict

from ovld import ovld

from . import data
from .map import (
    MapError,
    abstract_all,
    abstract_collect,
    abstract_map,
    abstract_map2,
)

################
# get_generics #
################


@abstract_collect.variant(
    initial_state=lambda: {"cache": {}, "prop": "$generics"}
)
def get_generics(self, gn: data.GenericBase):
    """Collect all generics."""
    return {gn}


#############
# canonical #
#############


@abstract_map.variant
def canonical(self, gn: data.GenericBase, *, mapping):
    """Return an AbstractValue with canonical generics.

    Essentially remaps the first encountered generic to CanonGeneric(0), the
    second to CanonGeneric(1), etc.

    The mapping from generic to canonical is stored in mapping.
    """
    if gn not in mapping:
        mapping[gn] = data.CanonGeneric(len(mapping))
    return mapping[gn]


###############
# uncanonical #
###############


@abstract_map.variant
def _uncanonical(self, gn: data.GenericBase, *, invmapping):
    """Undo the canonical transform."""
    return invmapping.get(gn, gn)


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
def fresh_generics(self, gn: data.GenericBase, *, mapping):
    """Map every generic to a fresh Placeholder."""
    if gn not in mapping:
        mapping[gn] = data.Placeholder()
    return mapping[gn]


#########
# reify #
#########


@abstract_all.variant
def is_concrete(self, _: data.GenericBase, **kwargs):
    """Return whether an AbstractValue contains any generics."""
    return False


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": None, "check": is_concrete}
)
def reify(self, gn: data.GenericBase, *, unif):
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

        if isinstance(cx, data.GenericBase):
            merged = cy
        elif isinstance(cy, data.GenericBase):
            merged = cx
        else:
            merged = ufn(cx, cy, U=self)

        eqv.add(merged)
        for entry in eqv:
            self.eqv[entry] = eqv
            self.canon[entry] = merged

        return merged


@abstract_map2.variant
def _unify(self, x: data.GenericBase, y: object, *, U):
    return U.unify(self, x, y)


@ovld
def _unify(self, x: object, y: data.GenericBase, *, U):  # noqa: F811
    return U.unify(self, x, y)


@ovld
def _unify(self, x: data.GenericBase, y: data.GenericBase, *, U):  # noqa: F811
    return U.unify(self, x, y)


@ovld
def _unify(self, x: object, y: object, *, U):  # noqa: F811
    if x == y:
        return x
    else:
        raise MapError(x, y, reason="Cannot merge objects")


@ovld
def _unify(  # noqa: F811
    self,
    x: data.AbstractUnion,
    y: (data.AbstractAtom, data.AbstractStructure),
    *,
    U
):
    """Check if abstract value is in abstrct union."""
    if y in x.options:
        return y
    else:
        raise MapError(x, y, reason="Cannot merge objects")


@ovld
def _unify(  # noqa: F811
    self,
    x: (data.AbstractAtom, data.AbstractStructure),
    y: data.AbstractUnion,
    *,
    U
):
    """Check if abstract value is in abstrct union."""
    if x in y.options:
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


@ovld
def _merge(self, x: data.ValueTrack, y: data.ValueTrack, *, U):  # noqa: F811
    """Return ANYTHING if values are different."""
    return x if x.value == y.value else data.ValueTrack(data.ANYTHING)


def merge(x, y, U=None):
    """Merge x with y using Unificator U.

    Works like unify except for AbstractUnion: merge(union1, union2) will
    append both unions' possibilities rather than unify them.
    """
    U = U or Unificator()
    res = _merge(x, y, U=U)
    return reify(res, unif=U.canon), U
