"""Test generic graph utilities."""

from myia.graph_utils import dfs


def _succ(x):
    return reversed(x) if isinstance(x, tuple) else ()


def test_dfs():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]


def test_dfs_dups():
    a = (1, 2, 2)
    b = (3, 4, 5, 2)
    c = (b, 6)
    d = (a, a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]
