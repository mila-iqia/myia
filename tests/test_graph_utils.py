"""Test generic graph utilities."""

from myia.graph_utils import dfs, toposort


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


def _check_toposort(order, root, succ):
    nodes = set(dfs(root, succ))
    assert len(order) == len(nodes)
    for node in nodes:
        for i in succ(node):
            assert order.index(i) < order.index(node)


def test_toposort():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    order = list(toposort(d, _succ))
    _check_toposort(order, d, _succ)


def test_toposort_overlap():
    a = (1, 2)
    b = (a, 2)
    c = (b, a, 2)

    order = list(toposort(c, _succ))
    _check_toposort(order, c, _succ)
