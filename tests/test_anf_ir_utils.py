from myia.anf_ir import Constant, Apply, Graph
from myia.anf_ir_utils import dfs, toposort, succ_incoming

from .test_graph_utils import _check_toposort


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
