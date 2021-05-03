from hrepr import pstr

from myia.abstract import data
from myia.abstract.data import ABSENT, ANYTHING
from myia.utils.intern import intern


def test_tracks():
    tr = data.Tracks(value=1, interface=int)
    assert tr.value == 1
    assert tr.interface is int
    vt = data.ValueTrack(1)
    it = data.InterfaceTrack(int)
    assert tr.get_track("value") == vt
    assert tr.get_track("interface") == it
    assert set(tr.values()) == {vt, it}
    assert dict(tr.items()) == {"value": vt, "interface": it}


def test_empty_tracks():
    tr = data.Tracks()
    assert tr.value is ANYTHING
    assert tr.interface is ABSENT


def test_AbstractAtom():
    s1 = data.AbstractAtom({"value": 1, "interface": int})
    s2 = data.AbstractAtom({"interface": int})

    assert s1.tracks.value == 1
    assert s1.tracks.interface is int

    assert s1.t.value == 1
    assert s1.t.interface is int

    assert s2.t.value is ANYTHING
    assert s2.t.interface is int


def makerec():
    i = data.AbstractAtom({"interface": int})
    rec = data.AbstractStructure.empty()
    un = data.AbstractUnion.empty()
    un.commit([i, rec], tracks={})
    rec.commit([un, un], tracks={"interface": tuple})
    return intern(rec)


def test_recursive():
    rec = makerec()
    rec2 = makerec()
    assert rec is rec2


def test_CanonGeneric():
    g1 = data.CanonGeneric(1)
    g2 = data.CanonGeneric(2)
    g1p = data.CanonGeneric(1)

    assert g1 == g1p
    assert g1 != g2

    assert {g1: 5}[g1p] == 5


def test_Placeholder():
    p1 = data.Placeholder()
    p2 = data.Placeholder()

    assert p1 != p2


def test_repr():
    i = data.AbstractAtom({"interface": int})
    assert str(i) == "*int()"

    i2 = data.AbstractAtom({"value": 7, "interface": int})
    assert str(i2) == "*int(value ↦ 7)"

    rec = makerec()
    assert str(rec) == "#1=*tuple(#2=*U(*int(), #1=*tuple()), #2=*U())"

    assert str(data.CanonGeneric(4)) == "?4"
    assert str(data.Placeholder()).startswith("??")


def test_repr_tracks():
    i = data.AbstractAtom({"interface": int, "value": 1})
    assert str(i) == "*int(value ↦ 1)"
    assert str(i.tracks) == "TrackDict(interface ↦ class int, value ↦ 1)"
    assert pstr(i.tracks, max_depth=0) == "<Tracks>"
    assert pstr(i.tracks, bare_tracks=True, max_depth=0) == ""


def test_repr_malformed():
    atm = data.AbstractAtom({})
    assert str(atm) == "AbstractAtom()"

    empt = data.AbstractStructure.empty()
    assert str(empt) == "AbstractStructure()"

    empt = data.AbstractUnion.empty()
    assert str(empt) == "AbstractUnion()"
