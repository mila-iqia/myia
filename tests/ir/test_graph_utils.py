from myia.ir.graph_utils import get_node_users, toposort
from myia.ir.print import _NodeCache
from myia.parser import parse
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


def test_toposort_factorial():
    nodecache = _NodeCache()
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


def test_node_users_factorial():
    nodecache = _NodeCache()
    g = parse_function(factorial)
    (param_n,) = g.parameters
    # Parameter n should be used twice: to test if n < 2, and passed to result of user_switch.
    assert (
        _str_list_nodes(nodecache, get_node_users(param_n))
        == """
#1 = #2(n)
#3 = _operator.lt(n, 2)
    """.strip()
    )
