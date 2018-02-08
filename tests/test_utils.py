import pytest

from myia.utils import Named, smap


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
