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


def test_clone():
    f = Graph()
    a = f.add_parameter("a")
    g = Graph()
    b = g.add_parameter("b")
    f.output = f.apply(g, a)
    g.output = g.apply("fma", b, 0, a)
    g.return_.add_edge(SEQ, g.output)

    g2 = g.clone()

    assert len(g2.parameters) == 1
    assert g2.return_
    assert g2.return_ is not g.return_
    assert g2.output.is_apply("fma")
    assert g2.output is not g.output

    fma2 = g2.output
    fma = g.output
    assert fma2.edges[0].node is not fma.edges[0].node
    assert fma2.edges[1].node is not fma.edges[1].node
    assert fma2.edges[2].node is fma.edges[2].node

    f2 = f.clone()
    f2.output.fn.value is f.output.fn.value

    op = f.apply("op", fma)
    op2 = op.clone({f: f})
    assert op2.edges[0].node is op.edges[0].node

    c = Constant(1)
    op3 = g.apply("op", c, c)
    op4 = op3.clone({g: g})
    assert op3.edges[0].node is not op4.edges[0].node
    assert op4.edges[0].node is op4.edges[1].node


def test_clone_closure():
    g = Graph()
    p = g.add_parameter("p")
    g2 = Graph(parent=g)
    g2.output = g2.apply("add", p, 1)
    g.output = g.apply("mul", g.apply(g2), g.apply(g2))

    h = g.clone()
    m = h.output
    h2 = m.edges[0].node.fn.value
    h2b = m.edges[1].node.fn.value
    assert h2 is not g2
    assert h2 is h2b
    assert h2.output.edges[0].node is h.parameters[0]


def test_graph_replace():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    a2 = g.apply("add", p, a)
    a2.add_edge(SEQ, a)
    a3 = g.apply("op", a2, a)
    a3.add_edge(SEQ, a2)
    g.output = a3

    g.delete_seq(a)
    g.replace_node(a, None, p)

    assert SEQ not in g.output.edges[SEQ].node.edges
    assert g.output.edges[1].node is p
    assert g.output.edges[0].node.edges[1].node is p


def test_graph_replace2():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    a2 = g.apply("add", p, a)
    a2.add_edge(SEQ, a)
    a3 = g.apply("op", a2, a)
    a3.add_edge(SEQ, a2)
    g.output = a3

    b = g.apply("make_int", p)

    g.replace_node(a2, SEQ, b)
    g.replace_node(a, None, b)

    assert g.output.edges[SEQ].node is b
    assert g.output.edges[1].node is b
    assert g.output.edges[0].node.edges[0].node is p


def test_graph_replace3():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    g2 = Graph(parent=g)
    g2.output = g2.apply("add", p, a)
    g.output = g.apply(g2)
    g.output.add_edge(SEQ, a)

    g.delete_seq(a)
    g.replace_node(a, None, p)

    assert g.output.fn.value.output.edges[1].node is p


def test_graph_replace4():
    g = Graph()
    p = g.add_parameter("p")
    a = g.apply("add", p, 0)
    g.output = a

    g.replace({(a, 0, p): Constant(1)})

    assert g.output.inputs[0].value == 1


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


def funfun(x):
    pass


def test_str():
    from myia.ir import print as irprint

    # str() uses a global labeler so that anonymous labels are consistent
    # from an execution to another, but that means prints occurring in other
    # tests could change the output, so we simply replace it by a new one
    # just for this test.
    # Note that this works because __hrepr_short__ imports the global_labeler
    # inside of the function

    glbl = irprint.global_labeler
    irprint.global_labeler = irprint.NodeLabeler()

    with enable_debug():
        c = Constant(1234)
        assert str(c) == "Constant(1234)"

        g = Graph()
        g.add_debug(name="g")
        assert str(g) == "Graph(g)"

        a = g.apply(funfun, c)
        assert str(a) == "Apply(#1 = test_node.funfun(1234))"

        b = g.apply(a, c)
        b.add_debug(name="b")
        assert str(b) == "Apply(b = #1(1234))"

        p = g.add_parameter("param")
        assert str(p) == "Parameter(param)"

    irprint.global_labeler = glbl


def test_hrepr():
    from hrepr import H, hrepr

    from myia.ir import print as irprint

    glbl = irprint.global_labeler
    irprint.global_labeler = irprint.NodeLabeler()

    def _hrepr(x, **kw):
        rval = hrepr(x, **kw)
        rval.resources = ()
        return rval

    with enable_debug():
        c = Constant(1234)
        assert _hrepr(c) == H.span["myia-Node", "myia-Constant"]("1234")

        g = Graph()
        g.add_debug(name="g")
        assert _hrepr(g, short=True) == H.span["myia-Graph"]("g")

        a = g.apply(funfun, c)
        assert _hrepr(a) == H.span["myia-Node", "myia-Apply"](
            "#1 = test_node.funfun(1234)"
        )

        b = g.apply(a, c)
        b.add_debug(name="b")
        assert _hrepr(b) == H.span["myia-Node", "myia-Apply"]("b = #1(1234)")

        p = g.add_parameter("param")
        assert _hrepr(p) == H.span["myia-Node", "myia-Parameter"]("param")

    irprint.global_labeler = glbl
