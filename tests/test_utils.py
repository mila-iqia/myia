import pytest

from myia.utils import Named, smap, HierDict


def test_named():
    named = Named('foo')
    assert repr(named) == 'foo'


def _sum(*args):
    return sum(args)


def test_smap():
    assert smap(_sum, 10, 20) == 30
    assert smap(_sum, 10, 20, 30, 40) == 100
    assert smap(_sum, (1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert smap(_sum, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert smap(_sum, [(1, [2]), 3], [(4, [5]), 6]) == [(5, [7]), 9]


def test_smap_failures():
    pairs = [
        ((1, 2), [1, 2]),
        ((1, 2), (1, 2, 3)),
        ((1, 2, 3), (1, 2)),
        (1, [1]),
        ([[1]], [1]),
    ]
    for a, b in pairs:
        with pytest.raises(TypeError):
            smap(_sum, a, b)


def test_HierDict():
    hd = HierDict(None)
    hd2 = HierDict(hd)

    hd['1'] = 1

    assert '1' not in hd2
    assert hd2['1'] == 1

    with pytest.raises(KeyError):
        hd['2']

    with pytest.raises(KeyError):
        hd2['2']
