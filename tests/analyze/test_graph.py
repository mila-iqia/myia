import pytest

from myia.ir.anf import ANFNode
from myia.analyze.graph import GraphAnalyzer, Frame


def test_Frame():
    f = Frame(None, ())
    assert len(f.types) == 0
    assert len(f.values) == 0

    f2 = Frame(f, ())
    assert len(f.types) == 0
    assert len(f.values) == 0

    f.types['1'] = 1

    assert '1' not in f2.types
    assert f2.types['1'] == 1

    f.values['2'] = 2

    assert '2' not in f2.values
    assert f2.values['2'] == 2

    f3 = Frame(f2, ())
    for i in range(1200):  # Above the python max stack depth
        f3 = Frame(f3, ())

    assert f3.types['1'] == 1
    assert f3.values['2'] == 2


def test_handle_node():
    ga = GraphAnalyzer()

    with pytest.raises(AssertionError):
        ga._handle_node(ANFNode((), None, None), Frame(None, ()))
