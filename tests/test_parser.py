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
