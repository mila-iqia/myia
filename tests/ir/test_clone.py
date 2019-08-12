
import pytest

from myia.debug.utils import GraphIndex
from myia.graph_utils import dfs
from myia.ir import (
    Constant,
    Graph,
    GraphCloner,
    GraphManager,
    Parameter,
    clone,
    succ_deeper,
    succ_incoming,
)
from myia.pipeline import scalar_parse as parse
from myia.prim import ops as P


def test_clone_simple():
    def f(x, y):
        a = x * x
        b = y * y
        c = a + b
        return c

    g = parse(f)

    cl = GraphCloner(g, clone_constants=True)

    g2 = cl[g]

    d1 = set(dfs(g.return_, succ_deeper))
    d2 = set(dfs(g2.return_, succ_deeper))

    # Both node sets should be disjoint
    assert d1 & d2 == set()

    # Without cloning constants
    cl2 = GraphCloner(g, clone_constants=False)

    g2 = cl2[g]

    d1 = set(dfs(g.return_, succ_deeper))
    d2 = set(dfs(g2.return_, succ_deeper))

    common = d1 & d2
    assert all(x.is_constant() for x in common)
    assert {x.value for x in common} == {P.scalar_add, P.scalar_mul, P.return_}


def test_clone_closure():
    def f(x, y):
        def j(z):
            a = x + y
            b = a + z
            return b
        c = j(3)
        return c

    parsed_f = parse(f)
    idx = GraphIndex(parsed_f)
    g = idx['j']

    cl = GraphCloner(g, clone_constants=True)
    idx2 = GraphIndex(cl[g], succ=succ_incoming)

    for name in 'xy':
        assert idx[name] is idx2[name]
    for name in 'zabj':
        assert idx[name] is not idx2[name]


def test_clone_scoping():
    def f(x, y):
        def g():
            # Depends on f, therefore cloned
            return x + y

        def h(z):
            # No dependency on f, so not nested and not cloned
            return z * z

        def i(q):
            # Depends on f, therefore cloned
            return g() * q
        return g() + h(x) + i(y)

    g = parse(f)

    cl = GraphCloner(g, clone_constants=True)

    g2 = cl[g]

    idx1 = GraphIndex(g)
    idx2 = GraphIndex(g2)

    for name in 'fgi':
        assert idx1[name] is not idx2[name]
    for name in 'h':
        assert idx1[name] is idx2[name]


def test_clone_total():
    def f1(x):
        return x * x

    def f2(y):
        return f1(y) + 3

    g = parse(f2)
    idx0 = GraphIndex(g)

    cl1 = GraphCloner(g, clone_constants=True, total=True)
    idx1 = GraphIndex(cl1[g])
    assert idx1['f2'] is not idx0['f2']
    assert idx1['f1'] is not idx0['f1']

    cl2 = GraphCloner(g, clone_constants=True, total=False)
    idx2 = GraphIndex(cl2[g])
    assert idx2['f2'] is not idx0['f2']
    assert idx2['f1'] is idx0['f1']


ONE = Constant(1)
TWO = Constant(2)
THREE = Constant(3)


def _graph_for_inline():
    target = Graph()
    target.debug.name = 'target'
    target.output = THREE
    return target


def _successful_inlining(cl, orig, new_params, target):
    assert cl[orig] is not target
    assert cl[orig] is orig

    new_root = cl[orig.output]
    assert new_root is not orig.output

    orig_nodes = set(dfs(orig.output, succ_incoming))
    new_nodes = set(dfs(new_root, succ_incoming))

    for p in new_params:
        assert p in new_nodes

    # Clones of orig's nodes should belong to target
    assert all(cl[node].graph in {target, None}
               for node in orig_nodes
               if node.graph is orig)

    # Clone did not change target
    assert target.output is THREE


def test_clone_inline():
    def f(x, y):
        a = x * x
        b = y * y
        c = a + b
        return c

    g = parse(f)

    target = _graph_for_inline()
    new_params = [ONE, TWO]
    cl = GraphCloner(inline=(g, target, new_params), clone_constants=False)

    _successful_inlining(cl, g, new_params, target)


def test_clone_recursive():
    def f(x, y):
        a = x * x
        b = y * y
        return f(a, b)

    g = parse(f)

    cl = GraphCloner(g, clone_constants=True)

    g2 = cl[g]

    d1 = set(dfs(g.return_, succ_deeper))
    d2 = set(dfs(g2.return_, succ_deeper))

    # Both node sets should be disjoint
    assert d1 & d2 == set()

    # Now test inlining
    target = _graph_for_inline()
    new_params = [ONE, TWO]
    cl2 = GraphCloner(inline=(g, target, new_params), clone_constants=True)

    _successful_inlining(cl2, g, new_params, target)

    # The recursive call still refers to the original graph
    new_nodes = set(dfs(cl2[g.output], succ_deeper))
    assert any(node.value is g for node in new_nodes)

    # Now test that inlining+total will fail
    target = _graph_for_inline()
    new_params = [ONE, TWO]
    with pytest.raises(Exception):
        cl2 = GraphCloner(inline=(g, target, new_params),
                          total=True,
                          clone_constants=True)


def test_clone_unused_parameters():
    @parse
    def f(x, y):
        return y

    cl = GraphCloner(f)
    f2 = cl[f]

    assert len(f2.parameters) == 2
    for p1, p2 in zip(f.parameters, f2.parameters):
        assert p1.graph is f
        assert p2.graph is f2
        assert cl[p1] is p2


def test_clone_without_forcing_manager():
    @clone
    @parse
    def f(x, y):
        return x * y

    clone(f)
    clone(f)
    GraphManager(f)


def test_clone_dangling_parameters():
    g = Graph()
    g.output = Parameter(g)

    g2 = clone(g)

    assert isinstance(g2.output, Parameter)
    assert g2.output.graph == g2
    assert len(g2.parameters) == 0
