import pytest

from myia.ir.node import SEQ, Constant, Graph, Node, Parameter
from myia.utils.info import enable_debug


def test_graph_output():
    g = Graph()

    with pytest.raises(ValueError):
        g.output

    c = g.constant(0)
    g.output = c

    assert g.output.is_constant() and g.output.value == 0

    c = g.constant(2)
    g.output = c

    assert g.output.is_constant() and g.output.value == 2


def test_graph_flags():
    g = Graph()

    assert "core" not in g.flags

    g.set_flags(core=True)

    assert g.flags["core"]

    g.set_flags(core=False)

    assert not g.flags["core"]


def test_parameter():
    g = Graph()

    p = g.add_parameter("a")
    assert p.is_parameter()
    assert p.graph is g
    assert p.name == "a"
    assert g.parameters[0] is p

    p2 = g.add_parameter("b")
    assert g.parameters[1] is p2


def test_apply():
    g = Graph()

    c = g.constant(0)

    a = g.apply("add", c, 1)

    assert a.is_apply()
    assert a.is_apply("add")
    assert a.graph is g
    assert isinstance(a.edges[1].node, Node)

    assert len(a.edges) == 3
    assert a.fn.value == "add"
    assert a.inputs == (c, a.edges[1].node)

    a2 = g.apply("sub", a, c)
    a2.add_edge(SEQ, a)

    assert a2.edges[SEQ].node is a

    with pytest.raises(AssertionError):
        a2.add_edge(SEQ, None)


def test_constant():
    g = Graph()

    c = g.constant(0)

    assert c.value == 0
    assert c.is_constant()
    assert not c.is_constant_graph()

    c2 = g.constant(g)

    assert c2.is_constant_graph()


def test_graph_replace():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    a2 = g.apply("add", p, a)
    a2.add_edge(SEQ, a)
    a3 = g.apply("op", a2, a)
    a3.add_edge(SEQ, a2)
    g.output = a3

    r = {a: p}
    r_s = {a: None}
    g.replace(r, r_s)

    assert g.output.edges[SEQ].node.edges[SEQ].node is None
    assert g.output.edges[1].node is p
    assert g.output.edges[0].node.edges[1].node is p


def test_graph_add_debug():
    g = Graph()

    g.add_debug(name="g", param=1)

    assert g.debug is None

    with enable_debug():
        g2 = Graph()

    g2.add_debug(name="g", param=1)

    assert g2.debug is not None
    assert g2.debug.name == "g"
    assert g2.debug.param == 1


def test_node():
    c = Constant(0)

    assert not c.is_apply()
    assert not c.is_parameter()

    c.add_debug(name="c", value=0)

    assert c.debug is None

    with enable_debug():
        p = Parameter(None, "a")

    assert not p.is_constant()
    assert not p.is_constant_graph()

    p.add_debug(floating=True)

    assert p.debug is not None
    assert p.debug.floating
