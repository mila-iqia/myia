from myia.abstract import data
from myia.abstract.map import abstract_all, abstract_any, abstract_map
from myia.utils.intern import intern


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


def makerec(value):
    i = data.AbstractAtom({"value": value, "interface": int})
    rec = data.AbstractStructure.empty()
    un = data.AbstractUnion([i, rec], tracks={})
    rec.commit([un, un], tracks={"interface": tuple})
    return intern(rec)


def test_rec():
    assert absolutify(makerec(-3)) is makerec(3)
    assert absolutify_noprop(makerec(-3)) is makerec(3)
    assert increment_value(makerec(-3)) is makerec(-2)
