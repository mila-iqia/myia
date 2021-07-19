import pytest
from ovld import ovld

from myia.abstract import data
from myia.abstract.map import (
    MapError,
    abstract_all,
    abstract_any,
    abstract_collect,
    abstract_map,
    abstract_map2,
)
from myia.testing.common import A, Un
from myia.utils.intern import intern

from ..common import one_test_per_assert


@abstract_any.variant
def hastup(self, x: data.InterfaceTrack):
    return x.value is tuple


@abstract_all.variant
def notup(self, x: data.InterfaceTrack):
    return x.value is not tuple


@abstract_map.variant
def increment_value(self, x: data.ValueTrack):
    return data.ValueTrack(x.value + 1)


def test_abspredicates():
    s1 = data.AbstractAtom({"value": 1, "interface": int})
    s2 = data.AbstractAtom({"value": 2, "interface": int})

    assert notup(s1)
    assert not hastup(s1)

    tu = data.AbstractStructure([s1, s2], tracks={"interface": tuple})

    assert not notup(tu)
    assert hastup(tu)

    li = data.AbstractStructure([s1, s2], tracks={"interface": list})

    assert notup(li)
    assert not hastup(li)

    un = data.AbstractUnion([s1, tu], tracks={})

    assert not notup(un)
    assert hastup(un)


def test_absmap():
    s1 = data.AbstractAtom({"value": 1, "interface": int})
    s2 = data.AbstractAtom({"value": 2, "interface": int})
    s3 = data.AbstractAtom({"value": 3, "interface": int})

    assert increment_value(s1) is s2
    assert increment_value(s2) is s3

    tu = data.AbstractStructure([s1, s2], tracks={"interface": tuple})

    tu_tr = data.AbstractStructure([s2, s3], tracks={"interface": tuple})

    assert increment_value(tu) is tu_tr

    un = data.AbstractUnion([s1, s2], tracks={})

    un_tr = data.AbstractUnion([s2, s3], tracks={})

    assert increment_value(un) is un_tr


@abstract_all.variant(initial_state=lambda: {"cache": {}, "prop": "_allpos"})
def allpos(self, x: data.ValueTrack):
    return x.value >= 0 or x.value == -1000


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": "_allpos", "check": allpos},
)
def absolutify(self, x: data.ValueTrack):
    return data.ValueTrack(abs(x.value))


@abstract_map.variant(
    initial_state=lambda: {"cache": {}, "prop": None, "check": allpos},
)
def absolutify_noprop(self, x: data.ValueTrack):
    return data.ValueTrack(abs(x.value))


def test_props():
    s2 = data.AbstractAtom({"value": 2, "interface": int})
    sm2 = data.AbstractAtom({"value": -2, "interface": int})

    assert absolutify(sm2) is s2

    tu = data.AbstractStructure([s2, sm2, sm2], tracks={"interface": tuple})

    tu_tr = data.AbstractStructure([s2, s2, s2], tracks={"interface": tuple})

    assert absolutify(tu) is tu_tr

    # This one is an inconsistency between allpos and absolutify, to check
    # that the transform is ignored if allpos returns True
    sm1000 = data.AbstractAtom({"value": -1000, "interface": int})
    assert absolutify(sm1000) is sm1000

    tu1000 = data.AbstractStructure(
        [sm1000, sm1000], tracks={"interface": tuple}
    )
    assert absolutify(tu1000) is tu1000


@abstract_map2.variant
def add(self, x: data.ValueTrack, y: data.ValueTrack):
    return data.ValueTrack(x.value + y.value)


@ovld
def add(  # noqa: F811
    self, x: data.InterfaceTrack, y: data.InterfaceTrack, **kwargs
):
    assert x.value == y.value
    return x


@one_test_per_assert
def test_map2():
    assert add(A(1), A(7)) is A(8)
    assert add(A(1, 2, 10), A(7, 0, 21)) is A(8, 2, 31)
    assert add(Un(1, 2, 10), Un(7, 0, 21)) is Un(8, 2, 31)


def test_map2_bad():
    with pytest.raises(MapError):
        add(A(1), A(9, 3))

    with pytest.raises(MapError):
        add(A(1, 2), A(1, 2, 3))

    with pytest.raises(MapError):
        add(Un(1, 2), Un(3, 4, 5))


def makerec(value):
    i = data.AbstractAtom({"value": value, "interface": int})
    rec = data.AbstractStructure.empty()
    un = data.AbstractUnion.empty()
    un.commit([i, rec], tracks={})
    rec.commit([un, un], tracks={"interface": tuple})
    return intern(rec)


@one_test_per_assert
def test_rec():
    assert absolutify(makerec(-3)) is makerec(3)
    assert absolutify_noprop(makerec(-3)) is makerec(3)
    assert increment_value(makerec(-3)) is makerec(-2)
    assert add(makerec(-1), makerec(4)) is makerec(3)


@abstract_collect.variant
def ints(self, x: data.Tracks):
    if x.interface is int:
        return {x.value}
    else:
        return set()


@abstract_collect.variant(
    initial_state=lambda: {"cache": {}, "prop": "$floats"}
)
def floats(self, x: data.Tracks):
    if x.interface is float:
        return {x.value}
    else:
        return set()


@one_test_per_assert
def test_abstract_collect():
    assert ints(A(1, 2, (3.5, 4))) == {1, 2, 4}
    assert ints(A(1)) == {1}
    assert ints(A(2.4, 8.9)) == set()
    assert ints(Un(2.4, 7, 8.9)) == {7}
    assert floats(A(1, 2, (3.5, 4))) == {data.ANYTHING}
    assert floats(A(2.4, 7, 8.9)) == {data.ANYTHING}
