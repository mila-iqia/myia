from myia.api import ENV, parse
from myia.debug.utils import GraphIndex
from myia.graph_utils import dfs as _dfs
from myia.ir import Apply, Constant, Graph, Parameter, Special, \
    accessible_graphs, destroy_disconnected_nodes, dfs, exclude_from_set, \
    freevars_boundary, is_apply, is_constant, is_constant_graph, \
    is_parameter, is_special, isomorphic, replace, succ_bidirectional, \
    succ_deep, succ_deeper, succ_incoming, toposort
from myia.parser import Parser
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
        if is_constant_graph(node):
            return node.value.debug.name or '.'
        elif isinstance(node, Constant):
            return str(node.value)
        else:
            return node.debug.name or '.'

    return set(map(name, nodes))


def test_dfs_variants():
    def f(x):
        z = x * x

        def g(y):
            return y + z
        w = z + 3
        q = g(w)
        return q

    graph = parse(f)
    inner_graph_ct, = [x for x in dfs(graph.return_) if is_constant_graph(x)]
    inner_graph = inner_graph_ct.value

    inner_ret = inner_graph.return_

    deep = _name_nodes(_dfs(inner_ret, succ_deep))
    assert deep == set('. return add y z mul x'.split())

    deeper = _name_nodes(_dfs(inner_ret, succ_deeper))
    assert deeper == set('. return add y z mul x w 3 q g'.split())

    _bound_fv = freevars_boundary(inner_graph, True)
    bound_fv = _name_nodes(_dfs(inner_ret, succ_deeper, _bound_fv))
    assert bound_fv == set('. return add y z'.split())

    _no_fv = freevars_boundary(inner_graph, False)
    no_fv = _name_nodes(_dfs(inner_ret, succ_deeper, _no_fv))
    assert no_fv == set('. return add y'.split())

    _excl_root = exclude_from_set([inner_ret])
    excl_root = _name_nodes(_dfs(inner_ret, succ_deeper, _excl_root))
    assert excl_root == set()


def test_disconnect():
    def exclude_ct_uses(x):
        if is_constant_graph(x) or x.graph is not None:
            return 'follow'
        else:
            return 'nofollow'

    def f(x):
        a = x * x
        _b = a + x  # Not connected to any output # noqa
        c = a * a
        d = c * c   # Connected to g's output

        def g(y):
            return d * y
        return g(c)

    g = Parser(ENV, f).parse(False)

    # Include {None} to get Constants (makes it easier to compare to live)
    cov = {None} | accessible_graphs(g)

    live = _name_nodes(_dfs(g.return_, succ_deep))
    assert live == set('x a mul c return . g d y'.split())

    total = _name_nodes(_dfs(g.return_,
                             succ_bidirectional(cov),
                             exclude_ct_uses))
    assert total == set('x a mul c return . g d y _b add'.split())

    destroy_disconnected_nodes(g)

    total2 = _name_nodes(_dfs(g.return_,
                              succ_bidirectional(cov),
                              exclude_ct_uses))
    assert total2 == live


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
    _check_isomorphic(idx1['inner1'], idx2['inner2'], False)


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
    assert is_constant(cg, Graph)
    assert is_constant_graph(cg)

    one = Constant(1)
    assert is_constant(one)
    assert is_constant(one, int)
    assert not is_constant(one, str)
    assert not is_constant_graph(one)

    a = Apply([cg, one], g)
    assert is_apply(a)

    p = Parameter(g)
    assert is_parameter(p)

    s = Special(1234, g)
    assert is_special(s)
    assert is_special(s, int)
    assert not is_special(s, str)


def test_replace():
    @parse
    def f1(x, y):
        a = x * y
        b = x + y
        c = a / (b * b)
        return c

    @parse
    def f2(x, y):
        a = x * y
        c = a / (10 * 10)
        return c

    @parse
    def f3(x, y):
        return 66

    idx = GraphIndex(f1)

    # Replacing inner node
    assert not isomorphic(f1, f2)
    replace(idx['b'], Constant(10))
    assert isomorphic(f1, f2)

    # Replacing output
    assert not isomorphic(f1, f3)
    replace(idx['c'], Constant(66))
    assert not isomorphic(f1, f2)
    assert isomorphic(f1, f3)
