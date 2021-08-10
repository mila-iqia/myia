import pytest

from myia.infer.infnode import infer_graph
from myia.ir.node import SEQ, Graph
from myia.ir.print import format_exc, str_graph
from myia.parser import parse
from myia.testing.common import A
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
  #3 = 'op'(None) ; type=NoneType
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


def add(x, y):
    return x + y


def fun(a, b):
    c = add(
        a,
        b,
    )
    return c * b


def add2(x, y):
    return x + y


def test_format_exc():
    with enable_debug():
        graph = parse(fun)
        with pytest.raises(Exception) as exc:
            infer_graph(graph, (A(int), A(None)))
        assert (
            format_exc(exc.value, mode="caret")
            == """File ./tests/ir/test_print.py, lines 89-92
In fun:clone(a~6::*int(), b::*NoneType())
89     c = add(
           ^^^^
90         a,
           ^^
91         b,
           ^^
92     )
       ^

File ./myia/basics.py, line 144
In add:clone(x~114::*int(), y~65::*NoneType())
144     return dict(item for kw in args for item in kw.items())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TypeError: No __radd__ method for <class 'NoneType'>"""
        )

        with pytest.raises(Exception) as exc:
            infer_graph(graph, (A(int), A(None)))
        assert (
            format_exc(exc.value, mode="color")
            == """File ./tests/ir/test_print.py, lines 89-92
In fun:clone(a~6::*int(), b::*NoneType())
89     c = \x1b[33m\x1b[1madd(\x1b[0m
90         \x1b[33m\x1b[1ma,\x1b[0m
91         \x1b[33m\x1b[1mb,\x1b[0m
92     \x1b[33m\x1b[1m)\x1b[0m

File ./myia/basics.py, line 144
In add:clone(x~114::*int(), y~65::*NoneType())
144     return \x1b[33m\x1b[1mdict(item for kw in args for item in kw.items())\x1b[0m

TypeError: No __radd__ method for <class 'NoneType'>"""
        )

    assert format_exc(TypeError("abc")) is None


def test_format_exc_no_debug():
    graph = parse(add2)
    with pytest.raises(Exception) as exc:
        infer_graph(graph, (A(int), A(None)))
    format_exc(exc.value, mode="caret")
    format_exc(exc.value, mode="color")
