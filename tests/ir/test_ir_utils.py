import pytest

from myia.debug.label import short_labeler
from myia.debug.utils import GraphIndex
from myia.graph_utils import dfs as _dfs
from myia.ir import (
    Apply,
    Constant,
    Graph,
    Parameter,
    Special,
    dfs,
    exclude_from_set,
    freevars_boundary,
    isomorphic,
    print_graph,
    print_node,
    succ_deep,
    succ_deeper,
    succ_incoming,
    toposort,
)
from myia.pipeline import scalar_parse as parse
from tests.test_graph_utils import _check_toposort


def test_dfs():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    assert next(dfs(value)) == value
    assert set(dfs(value)) == {value, in0, in1}


def test_dfs_graphs():
    g0 = Graph()
    in0 = Constant(g0)
    in1 = Constant(1)
    g0.return_ = in1
    value = Apply([in0], Graph())
    assert set(dfs(value)) == {value, in0}
    assert set(dfs(value, follow_graph=True)) == {value, in0, in1}


def test_toposort():
    g0 = Graph()
    g0.output = Constant(1)
    g1 = Graph()
    in0 = Constant(g0)
    value = Apply([in0], g1)
    g1.output = value

    order = list(toposort(g1.return_))
    _check_toposort(order, g1.return_, succ_incoming)


def test_toposort2():
    g0 = Graph()
    g0.output = Constant(33)
    g1 = Graph()
    in0 = Constant(g0)
    in1 = Constant(1)
    v1 = Apply([in0, in1], g1)
    v2 = Apply([in0, v1, in1], g1)
    g1.output = v2

    order = list(toposort(g1.return_))
    _check_toposort(order, g1.return_, succ_incoming)


def _name_nodes(nodes):
    def name(node):
        return short_labeler.label(node, fn_label=False)

    return set(map(name, nodes)) - {"·"}


def test_dfs_variants():
    def f(x):
        z = x * x

        def g(y):
            return y + z

        w = z + 3
        q = g(w)
        return q

    graph = parse(f)
    (inner_graph_ct,) = [x for x in dfs(graph.return_) if x.is_constant_graph()]
    inner_graph = inner_graph_ct.value

    inner_ret = inner_graph.return_

    deep = _name_nodes(_dfs(inner_ret, succ_deep))
    assert deep == set("return scalar_add y z scalar_mul x".split())

    deeper = _name_nodes(_dfs(inner_ret, succ_deeper))
    assert deeper == set("return scalar_add y z scalar_mul x w 3 q g".split())

    _bound_fv = freevars_boundary(inner_graph, True)
    bound_fv = _name_nodes(_dfs(inner_ret, succ_deeper, _bound_fv))
    assert bound_fv == set("return scalar_add y z".split())

    _no_fv = freevars_boundary(inner_graph, False)
    no_fv = _name_nodes(_dfs(inner_ret, succ_deeper, _no_fv))
    assert no_fv == set("return scalar_add y".split())

    _excl_root = exclude_from_set([inner_ret])
    excl_root = _name_nodes(_dfs(inner_ret, succ_deeper, _excl_root))
    assert excl_root == set()


def _check_isomorphic(g1, g2, expected=True):
    # Check that it works both ways
    assert isomorphic(g1, g2) == expected
    assert isomorphic(g2, g1) == expected


def test_isomorphic():
    @parse
    def f1(x, y):
        return x * y

    @parse
    def f2(a, b):
        return a * b

    @parse
    def f3(a, b):
        return a + b

    @parse
    def f4(x, y, z):
        return x * y

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, False)
    _check_isomorphic(f1, f4, False)
    _check_isomorphic(f4, f1, False)


def test_isomorphic_closures():
    @parse
    def f1(x):
        def inner1(y):
            return x * y

        return inner1(10)

    @parse
    def f2(a):
        def inner2(b):
            return a * b

        return inner2(10)

    @parse
    def f3(a):
        def inner3(b):
            return a + b

        return inner3(10)

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, False)

    idx1 = GraphIndex(f1)
    idx2 = GraphIndex(f2)

    # inner1 and inner2 are not considered isomorphic because their free
    # variables are not matched together. They are matched together when we
    # check from f1 and f2, but not when we start from the closures directly.
    _check_isomorphic(idx1["inner1"], idx2["inner2"], False)


def test_isomorphic_globals():
    def helper1(x):
        return x * x

    def helper2(a):
        return a * a

    def helper3(a):
        return a

    @parse
    def f1(x):
        return helper1(x) * helper1(4)

    @parse
    def f2(a):
        return helper1(a) * helper1(4)

    @parse
    def f3(a):
        return helper2(a) * helper1(4)

    @parse
    def f4(a):
        return helper2(a) * helper3(4)

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, True)
    _check_isomorphic(f1, f4, False)


def test_isomorphic_recursion():
    def f1(x):
        if x < 0:
            return x
        else:
            return f1(x - 1)

    def f2(x):
        if x < 0:
            return x
        else:
            return f2(x - 1)

    def f3(x):
        if x > 0:
            return x
        else:
            return f3(x - 1)

    f1 = parse(f1)
    f2 = parse(f2)
    f3 = parse(f3)

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, False)


def test_helpers():
    g = Graph()
    cg = Constant(g)
    assert cg.is_constant(Graph)
    assert cg.is_constant_graph()

    one = Constant(1)
    assert one.is_constant()
    assert one.is_constant(int)
    assert not one.is_constant(str)
    assert not one.is_constant_graph()

    a = Apply([cg, one], g)
    assert a.is_apply()

    p = Parameter(g)
    assert p.is_parameter()

    s = Special(1234, g)
    assert s.is_special()
    assert s.is_special(int)
    assert not s.is_special(str)


def test_print_graph():
    g = Graph()
    g.debug.name = "testfn"
    p = g.add_parameter()
    p.debug.name = "a"
    p2 = g.add_parameter()
    p2.debug.name = "b"
    c = g.constant(1)
    g.output = g.apply(g, c, p2)
    g.output.debug.name = "_apply0"

    s = print_graph(g)
    assert (
        s
        == """graph testfn(%a, %b) {
  %_apply0 = @testfn(1, %b)
  return %_apply0
}
"""
    )

    # We use fake types here because we only care about the printing logic,
    # not the proper use of types.
    p.abstract = "int64"
    p2.abstract = "float32"
    g.output.abstract = "bool"
    g.return_.abstract = "bool"

    s = print_graph(g)
    assert (
        s
        == """graph testfn(%a : int64, %b : float32) -> bool {
  %_apply0 = @testfn(1, %b) ; type=bool
  return %_apply0
}
"""
    )


def test_print_node():
    g = Graph()
    g.debug.name = "testfn"
    p = g.add_parameter()
    p.debug.name = "a"
    p2 = g.add_parameter()
    p2.debug.name = "b"
    c = g.constant(1)
    g.output = g.apply(g, c, p2)
    g.output.debug.name = "_apply0"
    s = print_node(g.output)
    assert s == "%_apply0 = @testfn(1, %b)\n"


def test_print_cycle():
    g = Graph()
    g.debug.name = "testfn"
    p = g.add_parameter()
    p.debug.name = "a"
    p2 = g.add_parameter()
    p2.debug.name = "b"
    node = g.apply("make_tuple", p)
    node2 = g.apply("make_tuple", p2, node)
    node.inputs.append(node2)
    g.output = node2

    with pytest.raises(ValueError, match="cycle"):
        print_graph(g, allow_cycles=False)

    print_graph(g)


def test_print_closure():
    g = Graph()
    g.debug.name = "testfn"
    p = g.add_parameter()
    p.debug.name = "a"
    n = g.apply("make_tuple", p, p)
    n.debug.name = "_apply0"
    g2 = Graph()
    g2.debug.name = "sub"
    n2 = g2.apply("tuple_getitem", n, 0)
    n2.debug.name = "_apply1"
    g2.output = n2
    g.output = g.apply(g2)

    s = print_graph(g2)

    assert (
        s
        == """graph sub() {
  %_apply1 = tuple_getitem(%_apply0, 0)
  return %_apply1
}
"""
    )
    s = print_node(n2)

    assert s == "%_apply1 = tuple_getitem(%_apply0, 0)\n"
