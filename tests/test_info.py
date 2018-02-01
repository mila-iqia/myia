
import pytest
from myia.info import DebugInfo, DebugInherit, NamedDebugInfo


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
    class O: pass
    o = O()
    d = NamedDebugInfo(o)
    assert d.obj is o
    del o
    assert d.obj is None
