import operator as opr

import pytest

from myia.ir.graph_utils import (
    EXCLUDE,
    FOLLOW,
    NOFOLLOW,
    dfs,
    succ_deep,
    succ_deeper,
    succ_incoming,
    toposort,
)
from myia.ir.print import NodeLabeler
from myia.parser import parse
from myia.testing.common import build_graph
from myia.utils.info import enable_debug


def parse_function(function):
    with enable_debug():
        graph = parse(function)
    return graph


def _str_node(nodecache, n):
    if n.is_apply():
        return f"{nodecache(n)} = {nodecache(n.fn)}({', '.join(nodecache(i) for i in n.inputs)})"
    elif n.is_constant():
        return f"{type(n.value).__name__} {nodecache(n)}"
    else:
        return nodecache(n)


def _str_list_nodes(nodecache, nodes):
    return "\n".join(_str_node(nodecache, n) for n in nodes)


def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


def _succ(x):
    return reversed(x) if isinstance(x, tuple) else ()


def test_dfs():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    order = list(dfs(d, _succ))

    assert order == [d, a, 1, 2, c, b, 3, 4, 5, 6]


def test_dfs_complex_policy():
    a = (1, 2)
    b = (3, 4, 5)
    c = (b, 6)
    d = (a, c)

    def inc(n):
        if n is a:
            return EXCLUDE
        elif n is b:
            return NOFOLLOW
        else:
            return FOLLOW

    order = list(dfs(d, _succ, include=inc))

    assert order == [d, c, b, 6]


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


def test_succ():
    g1, nodeset1 = build_graph((opr.add, 1, (opr.sub, 2, "x")), params="x")
    g2, nodeset2 = build_graph((g1, 8, "y"), params="y")
    g3, nodeset3 = build_graph((g1, 9, "a"), params="abc")

    assert set(dfs(g1.return_, succ=succ_incoming)) == nodeset1
    assert set(dfs(g2.return_, succ=succ_incoming)) == nodeset2

    assert set(dfs(g1.return_, succ=succ_deep)) == nodeset1
    assert set(dfs(g2.return_, succ=succ_deep)) == nodeset1 | nodeset2

    assert set(dfs(g2.output, succ=succ_deeper)) == nodeset1 | nodeset2
    assert set(dfs(g3.return_, succ=succ_deeper)) == nodeset1 | nodeset3


def test_toposort_factorial():
    nodecache = NodeLabeler()
    g = parse_function(factorial)
    nodes = toposort(g)
    assert (
        _str_list_nodes(nodecache, nodes)
        == """
builtin_function_or_method _operator.lt
int 2
n
builtin_function_or_method _operator.truth
#1 = _operator.lt(n, 2)
function myia.basics.user_switch
Graph factorial:if_false
Graph factorial:if_true
#2 = _operator.truth(#1)
#3 = myia.basics.user_switch(#2, factorial:if_true, factorial:if_false)
function myia.basics.return_
#4 = #3(n)
#5 = myia.basics.return_(#4)
""".strip()
    )

    # Make sure any apply node comes after its fn and inputs.
    seen = set()
    for n in nodes:
        if n.is_apply():
            assert n.fn in seen
            for inp in n.inputs:
                assert inp in seen
        seen.add(n)
