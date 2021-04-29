import pytest

from myia.ir.node import SEQ, Graph
from myia.ir.print import str_graph
from myia.utils.info import enable_debug


def test_print1():
    g = Graph()
    p = g.add_parameter("p")
    p.abstract = "ptype"
    g.output = g.apply("op", None)
    g.output.abstract = "NoneType"
    g.return_.abstract = "NoneType"

    assert (
        str_graph(g)
        == """graph #1(#2: ptype) -> NoneType {
  #3 = op(None) ; type=NoneType
  return #3
}
"""
    )


def test_print2():
    g = Graph()
    f = Graph()
    g.output = g.apply(g.constant(f))
    g.return_.add_edge(SEQ, g.output)
    f.output = f.apply(g.constant(g))
    f.return_.add_edge(SEQ, f.output)

    assert (
        str_graph(g, recursive=True)
        == """graph #1() {
  #2 = #3()
  return #2
}

graph #3() {
  #4 = #1()
  return #4
}
"""
    )


def test_print3():
    g = Graph()
    a = g.apply("op")
    g.output = g.apply("op2")
    g.return_.add_edge(SEQ, g.output)
    g.output.add_edge(SEQ, a)
    a.add_edge(SEQ, g.output)

    with pytest.raises(ValueError):
        str_graph(g)

    str_graph(g, allow_cycles=True)


def test_print4():
    g = Graph()
    p = g.add_parameter("p")
    g.output = p
    g.return_.add_edge(SEQ, p)

    str_graph(g, allow_cycles=True)


def test_print5():
    with enable_debug():
        g = Graph()
        g.add_parameter("p")
        g.output = g.add_parameter("p")

    str_graph(g)
