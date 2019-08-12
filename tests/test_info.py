from threading import Thread

from myia.info import About, DebugInfo, DebugInherit, NamedDebugInfo


def test_nested_info():
    """Test DebugInherit as context manager."""
    with DebugInherit(a=1):
        with DebugInherit(b=2):
            with DebugInherit(c=3):
                info = DebugInfo(d=4)
                assert info.a == 1
                assert info.b == 2
                assert info.c == 3
                assert info.d == 4
    info = DebugInfo(d=4)
    for attr in 'abc':
        assert not hasattr(info, attr)
    assert info.d == 4


def test_info_trace():
    """Test that NamedDebugInfo saves a trace."""
    d = NamedDebugInfo()
    assert d.trace is None
    with DebugInherit(save_trace=True):
        d = NamedDebugInfo()
        assert d.trace is not None


def test_info_obj():
    """Test that NamedDebugInfo only holds a weak reference to its object."""
    class Ob:
        pass
    o = Ob()
    d = NamedDebugInfo(o)
    assert d.obj is o
    del o
    assert d.obj is None


def test_info_find():
    a = NamedDebugInfo()
    a.field1 = 1
    a.field2 = 2
    with About(a, 'thing'):
        b = NamedDebugInfo()
        b.field2 = 3
    assert a.find('field1') == 1
    assert a.find('field2') == 2
    assert b.find('field1') == 1
    assert b.find('field2') == 3
    assert b.find('field3') is None


def test_info_thread():
    exc = None

    def f():
        nonlocal exc
        try:
            a = NamedDebugInfo()
            with About(a, 'thing'):
                NamedDebugInfo()
        except Exception as e:
            exc = e

    t = Thread(target=f)
    t.start()
    t.join()
    if exc:
        raise exc
