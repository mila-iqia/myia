
import pytest
from myia.info import DebugInfo


def test_nested_info():
    """Test DebugInfo as context manager."""
    with DebugInfo(a=1):
        with DebugInfo(b=2):
            with DebugInfo(c=3):
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
    """Test that DebugInfo saves a trace."""
    d = DebugInfo()
    assert d.trace is None
    with DebugInfo(save_trace=True):
        d = DebugInfo()
        assert d.trace is not None


def test_info_obj():
    """Test that DebugInfo only holds a weak reference to its object."""
    class O: pass
    o = O()
    d = DebugInfo(o)
    assert d.obj is o
    del o
    assert d.obj is None
