"""Test generic graph utilities."""

import pytest

from myia.graph_utils import EXCLUDE, FOLLOW, NOFOLLOW, dfs, toposort


def _succ(x):
    return reversed(x) if isinstance(x, tuple) else ()


def _incl(x):
    return FOLLOW


def test_dfs():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]


def test_dfs_bad_include():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    def inc(n):
        return None

    with pytest.raises(ValueError):
        list(dfs(d, _succ, inc))


def test_dfs_dups():
    a = (1, 2, 2)
    b = (3, 4, 5, 2)
    c = (b, 6)
    d = (a, a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]


def _check_toposort(order, root, succ=_succ, incl=_incl):
    nodes = set(dfs(root, succ, incl))
    assert set(order) == nodes
    for node in nodes:
        for i in succ(node):
            if i in order:
                assert order.index(i) < order.index(node)


def test_toposort():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, a, 6)
    d = (a, c)

    order = list(toposort(d, _succ, _incl))
    _check_toposort(order, d, _succ, _incl)


def test_toposort_bad_include():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, a, 6)
    d = (a, c)

    def inc(n):
        return None

    with pytest.raises(ValueError):
        list(toposort(d, _succ, inc))


def test_toposort_overlap():
    a = (1, 2)
    b = (a, 2)
    c = (b, a, 2)

    order = list(toposort(c, _succ, _incl))
    _check_toposort(order, c, _succ, _incl)


def test_toposort_incl():
    def _incl_nf(x):
        if isinstance(x, tuple) and len(x) == 2:
            return NOFOLLOW
        else:
            return FOLLOW

    def _incl_x(x):
        if isinstance(x, tuple) and len(x) == 2:
            return EXCLUDE
        else:
            return FOLLOW

    a = (1, 2)
    b = (a, 3)
    c = (a, b, 4, 5)

    order1 = list(toposort(c, _succ, _incl_nf))
    _check_toposort(order1, c, _succ, _incl_nf)

    order2 = list(toposort(c, _succ, _incl_x))
    _check_toposort(order2, c, _succ, _incl_x)


def test_toposort_cycle():
    class Q:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def qsucc(q):
        return [q.x, q.y]

    q = Q(1, 2)
    q.y = q

    with pytest.raises(ValueError):
        list(toposort(q, qsucc, _incl))
