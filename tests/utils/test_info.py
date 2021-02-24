from threading import Thread

import pytest

from myia.utils.info import (
    AbbreviationTranslator,
    DebugInfo,
    Labeler,
    about,
    attach_debug_info,
    debug_inherit,
)


class Ob:
    pass


def test_nested_info():
    """Test debug_inherit as context manager."""
    with debug_inherit(a=1):
        with debug_inherit(b=2):
            with debug_inherit(c=3):
                info = DebugInfo(d=4)
                assert info.a == 1
                assert info.b == 2
                assert info.c == 3
                assert info.d == 4
    info = DebugInfo(d=4)
    for attr in "abc":
        assert not hasattr(info, attr)
    assert info.d == 4


def test_info_trace():
    """Test that DebugInfo saves a trace."""
    d = DebugInfo()
    assert d.trace is None
    with debug_inherit(save_trace=True):
        d = DebugInfo()
        assert d.trace is not None


def test_info_obj():
    """Test that DebugInfo only holds a weak reference to its object."""
    o = Ob()
    d = DebugInfo(o)
    assert d.obj is o
    del o
    assert d.obj is None


def test_info_find():
    a = DebugInfo()
    a.field1 = 1
    a.field2 = 2
    with about(a, "thing"):
        b = DebugInfo()
        b.field2 = 3
    assert a.find("field1") == 1
    assert a.find("field2") == 2
    assert b.find("field1") == 1
    assert b.find("field2") == 3
    assert b.find("field3") is None


def test_about():
    o = Ob()
    info = DebugInfo(o, a=1, b=2)

    with pytest.raises(TypeError):
        with about(o, "shape"):
            pass

    with about(info, "shape"):
        o2 = Ob()
        info2 = DebugInfo(o2)
        assert info2.about is info
        assert info2.relation == "shape"


def test_attach_debug_info():
    o = Ob()
    assert attach_debug_info(o, a=1, b=2) is o

    with about(o, "married"):
        p = Ob()
        attach_debug_info(p, a=7)

    assert o.debug.a == 1
    assert o.debug.b == 2

    assert p.debug.a == 7
    assert not hasattr(p.debug, "b")
    assert p.debug.about is o.debug
    assert p.debug.relation == "married"


def test_info_thread():
    exc = None

    def f():
        nonlocal exc
        try:
            a = DebugInfo()
            with about(a, "thing"):
                DebugInfo()
        except Exception as e:
            exc = e

    t = Thread(target=f)
    t.start()
    t.join()
    if exc:
        raise exc


class Auto:
    def __init__(self, name=None, value=None):
        self.value = value
        attach_debug_info(self, name=name)


@pytest.fixture
def manyinfo():
    rval = Ob()
    rval.x1 = Auto(name="toyota")
    with about(rval.x1, "apple"):
        rval.x2 = Auto()
        rval.x2b = Auto()
        rval.x2c = Auto()
        with about(rval.x2, "blueberry"):
            rval.x3 = Auto()
            with about(rval.x3, "corn"):
                rval.x4 = Auto()
                rval.x4b = Auto()
    return rval


def test_labeler(manyinfo):
    m = manyinfo
    lbl = Labeler()
    assert lbl(m.x1) == "toyota"
    assert lbl(m.x1.debug) == "toyota"
    assert lbl(m.x2) == "apple:toyota"
    assert lbl(m.x2c) == "apple:toyota//2"
    assert lbl(m.x2b) == "apple:toyota//3"
    assert lbl(m.x3) == "apple:blueberry:toyota"
    assert lbl(m.x4) == "apple:blueberry:corn:toyota"
    assert lbl(m.x4b) == "apple:blueberry:corn:toyota//2"


def test_labeler_abbrv(manyinfo):
    m = manyinfo
    tr = AbbreviationTranslator(
        {
            "apple": "&",
            "corn": "#",
        }
    )
    lbl = Labeler(relation_translator=tr)
    assert lbl(m.x1) == "toyota"
    assert lbl(m.x2) == "&toyota"
    assert lbl(m.x2c) == "&toyota//2"
    assert lbl(m.x2b) == "&toyota//3"
    assert lbl(m.x3) == "&blueberry:toyota"
    assert lbl(m.x4) == "&blueberry:#toyota"
    assert lbl(m.x4b) == "&blueberry:#toyota//2"


def test_labeler_nonames():
    x1 = Auto()
    with about(x1, "assassin"):
        x2 = Auto()
    x3 = Auto()

    lbl = Labeler()
    assert lbl(x1) == "#1"
    assert lbl(x1.debug) == "#1"
    assert lbl(x2) == "assassin:#1"
    assert lbl(x3) == "#2"


def test_labeler_object_describer():
    def descr(obj):
        return obj.value and str(obj.value)

    x1 = Auto(name="bob", value=12)

    lbl = Labeler(object_describer=descr)
    assert lbl(x1) == "12"