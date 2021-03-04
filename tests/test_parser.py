import pytest

from myia.parser import parse
from myia.ir.print import str_graph


def test_simple():
    def f(x):
        return x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = load(x, 1)
  return %_apply0
}
"""


def test_closure():
    x = 1
    def f():
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = load(x, 0)
  return %_apply0
}
"""


def test_add():
    def f(x, y):
        return x + y

    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = load(x, 2)
  %_apply1 = load(y, 3)
  %_apply2 = add(%_apply0, %_apply1)
  return %_apply2
}
"""


def test_seq():
    def f(x):
        x = x + 1
        return 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = load(x, 1)
  %_apply1 = add(%_apply0, 1)
  %_apply2 = store(x, %_apply1, 2)
  return 0
}
"""

def test_ifexp():
    def f(x, y, b):
        return x if b else y

    assert str_graph(parse(f)) == """graph f(%x, %y, %b) {
  %_apply0 = load(b, 3)
  %_apply1 = user_switch(%_apply0, @if_true, @if_false)
  %_apply2 = %_apply1()
  return %_apply2
}

graph if_false() {
  %_apply3 = load(y, 0)
  return %_apply3
}

graph if_true() {
  %_apply4 = load(x, 0)
  return %_apply4
}
"""

def test_boolop():
    def f(a, b, c):
        return a and b or c

    assert str_graph(parse(f)) == """graph f(%a, %b, %c) {
  %_apply0 = load(a, 3)
  %_apply1 = switch(%_apply0, @if_true, @if_false)
  %_apply2 = %_apply1()
  %_apply3 = switch(%_apply2, @if_true, @if_false)
  %_apply4 = %_apply3()
  return %_apply4
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply5 = load(b, 0)
  return %_apply5
}

graph if_false() {
  %_apply6 = load(c, 0)
  return %_apply6
}

graph if_true() {
  return True
}
"""


def test_compare():
    def f(x):
        return 0 < x < 42

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = load(x, 1)
  %_apply1 = lt(0, %_apply0)
  %_apply2 = switch(%_apply1, @if_true, @if_false)
  %_apply3 = %_apply2()
  return %_apply3
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply4 = load(x, 0)
  %_apply5 = lt(%_apply4, 42)
  return %_apply5
}
"""

def test_unary():
    def f(x):
        return -x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = load(x, 1)
  %_apply1 = neg(%_apply0)
  return %_apply1
}
"""
