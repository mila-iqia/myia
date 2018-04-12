"""Test generic graph utilities."""

from myia.graph_utils import dfs, toposort, FOLLOW, NOFOLLOW, EXCLUDE


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


def test_dfs_dups():
    a = (1, 2, 2)
    b = (3, 4, 5, 2)
    c = (b, 6)
    d = (a, a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]


def _check_toposort(order, root, succ=_succ, incl=_incl):
    nodes = set(dfs(root, succ, incl))
    assert len(order) == len(nodes)
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
