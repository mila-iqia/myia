import pytest

from myia.parser import parse
from myia.ir.print import str_graph


def test_simple():
    def f(x):
        return x

    assert str_graph(parse(f)) == """graph f(%x) {
  return %x
}
"""


def test_free():
    x = 1
    def f():
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""


def test_global():
    x = 1
    def f():
        global x
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""


def Xtest_nonlocal():
    def f():
        x = 1
        def g():
            nonlocal x
            x = x + 1
        g()
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = load(x)
  %_apply1 = add(%_apply0, 1)
  %_apply2 = store(x, %_apply1)
  %_apply3 = load(x)
  return %_apply3
}
"""


def test_add():
    def f(x, y):
        return x + y
    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = add(%x, %y)
  return %_apply0
}
"""


def test_seq():
    def f(x):
        x = x + 1
        return 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = add(%x, 1)
  return 0
}
"""


def test_seq2():
    def f(x):
        return x + x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = add(%x, %x)
  return %_apply0
}
"""


def test_ifexp():
    def f(x, y, b):
        return x if b else y

    assert str_graph(parse(f)) == """graph f(%x, %y, %b) {
  %_apply0 = store(x, %x)
  %_apply1 = store(y, %y)
  %_apply2 = store(b, %b)
  %_apply3 = load(b)
  %_apply4 = user_switch(%_apply3, @if_true, @if_false)
  %_apply5 = %_apply4()
  return %_apply5
}

graph if_false() {
  %_apply6 = load(y)
  return %_apply6
}

graph if_true() {
  %_apply7 = load(x)
  return %_apply7
}
"""

def test_boolop():
    def f(a, b, c):
        return a and b or c

    assert str_graph(parse(f)) == """graph f(%a, %b, %c) {
  %_apply0 = store(a, %a)
  %_apply1 = store(b, %b)
  %_apply2 = store(c, %c)
  %_apply3 = load(a)
  %_apply4 = switch(%_apply3, @if_true, @if_false)
  %_apply5 = %_apply4()
  %_apply6 = switch(%_apply5, @if_true, @if_false)
  %_apply7 = %_apply6()
  return %_apply7
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply8 = load(b)
  return %_apply8
}

graph if_false() {
  %_apply9 = load(c)
  return %_apply9
}

graph if_true() {
  return True
}
"""

def test_compare():
    def f(x):
        return x > 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = gt(%x, 0)
  return %_apply0
}
"""


def test_compare2():
    def f(x):
        return 0 < x < 42

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = store(x, %x)
  %_apply1 = load(x)
  %_apply2 = lt(0, %_apply1)
  %_apply3 = switch(%_apply2, @if_true, @if_false)
  %_apply4 = %_apply3()
  return %_apply4
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply5 = load(x)
  %_apply6 = lt(%_apply5, 42)
  return %_apply6
}
"""

def test_unary():
    def f(x):
        return -x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = neg(%x)
  return %_apply0
}
"""


def test_if():
    def f(b, x, y):
        if b:
            return x
        else:
            return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = store(b, %b)
  %_apply1 = store(x, %x)
  %_apply2 = store(y, %y)
  %_apply3 = load(b)
  %_apply4 = user_switch(%_apply3, @if_true, @if_false)
  %_apply5 = %_apply4()
  return %_apply5
}

graph if_false() {
  %_apply6 = load(y)
  return %_apply6
}

graph if_true() {
  %_apply7 = load(x)
  return %_apply7
}
"""

def test_if2():
    def f(b, x, y):
        if b:
            return x
        return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = store(b, %b)
  %_apply1 = store(x, %x)
  %_apply2 = store(y, %y)
  %_apply3 = load(b)
  %_apply4 = user_switch(%_apply3, @if_true, @if_false)
  %_apply5 = %_apply4()
  return %_apply5
}

graph if_false() {
  %_apply6 = @if_after()
  return %_apply6
}

graph if_after() {
  %_apply7 = load(y)
  return %_apply7
}

graph if_true() {
  %_apply8 = load(x)
  return %_apply8
}
"""


def test_while():
    def f(b, x, y):
        while b:
            return x
        return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = store(b, %b)
  %_apply1 = store(x, %x)
  %_apply2 = store(y, %y)
  %_apply3 = @while_header()
  return %_apply3
}

graph while_header() {
  %_apply4 = load(b)
  %_apply5 = user_switch(%_apply4, @while_body, @while_after)
  %_apply6 = %_apply5()
  return %_apply6
}

graph while_after() {
  %_apply7 = load(y)
  return %_apply7
}

graph while_body() {
  %_apply8 = load(x)
  return %_apply8
}
"""


def test_while2():
    def f(x):
        while x:
            x = x - 1
        return x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = store(x, %x)
  %_apply1 = @while_header()
  return %_apply1
}

graph while_header() {
  %_apply2 = load(x)
  %_apply3 = user_switch(%_apply2, @while_body, @while_after)
  %_apply4 = %_apply3()
  return %_apply4
}

graph while_after() {
  %_apply5 = load(x)
  return %_apply5
}

graph while_body() {
  %_apply6 = load(x)
  %_apply7 = sub(%_apply6, 1)
  %_apply8 = store(x, %_apply7)
  %_apply9 = @while_header()
  return %_apply9
}
"""
