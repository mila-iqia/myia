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
    def f():
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""


def test_global():
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
  %_apply1 = <built-in function add>(%_apply0, 1)
  %_apply2 = store(x, %_apply1)
  %_apply3 = load(x)
  return %_apply3
}
"""


def test_add():
    def f(x, y):
        return x + y
    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = <built-in function add>(%x, %y)
  return %_apply0
}
"""


def test_not_in():
    def f(x, y):
        return x not in y

    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = <built-in function contains>(%x, %y)
  %_apply1 = <built-in function not_>(%_apply0)
  return %_apply1
}
"""


def test_seq():
    def f(x):
        x = x + 1
        return 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function add>(%x, 1)
  return 0
}
"""


def test_seq2():
    def f(x):
        return x + x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function add>(%x, %x)
  return %_apply0
}
"""


def test_ifexp():
    def f(x, y, b):
        return x if b else y

    assert str_graph(parse(f)) == """graph f(%x, %y, %b) {
  %_apply0 = universe_setitem(%_apply1, %x)
  %_apply2 = universe_setitem(%_apply3, %y)
  %_apply4 = <built-in function truth>(%b)
  %_apply5 = user_switch(%_apply4, @if_true, @if_false)
  %_apply6 = %_apply5()
  return %_apply6
}

graph if_false() {
  %_apply7 = universe_getitem(%_apply3)
  return %_apply7
}

graph if_true() {
  %_apply8 = universe_getitem(%_apply1)
  return %_apply8
}
"""

def test_boolop():
    def f(a, b, c):
        return a and b or c

    assert str_graph(parse(f)) == """graph f(%a, %b, %c) {
  %_apply0 = universe_setitem(%_apply1, %b)
  %_apply2 = universe_setitem(%_apply3, %c)
  %_apply4 = <built-in function truth>(%a)
  %_apply5 = switch(%_apply4, @if_true, @if_false)
  %_apply6 = %_apply5()
  %_apply7 = <built-in function truth>(%_apply6)
  %_apply8 = switch(%_apply7, @if_true, @if_false)
  %_apply9 = %_apply8()
  return %_apply9
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply10 = universe_getitem(%_apply1)
  return %_apply10
}

graph if_false() {
  %_apply11 = universe_getitem(%_apply3)
  return %_apply11
}

graph if_true() {
  return True
}
"""

def test_compare():
    def f(x):
        return x > 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function gt>(%x, 0)
  return %_apply0
}
"""


def test_compare2():
    def f(x):
        return 0 < x < 42

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = universe_setitem(%_apply1, %x)
  %_apply2 = universe_getitem(%_apply1)
  %_apply3 = <built-in function lt>(0, %_apply2)
  %_apply4 = <built-in function truth>(%_apply3)
  %_apply5 = switch(%_apply4, @if_true, @if_false)
  %_apply6 = %_apply5()
  return %_apply6
}

graph if_false() {
  return False
}

graph if_true() {
  %_apply7 = universe_getitem(%_apply1)
  %_apply8 = <built-in function lt>(%_apply7, 42)
  return %_apply8
}
"""


def test_lambda():
    def f():
        l = lambda x: x
        return l

    assert str_graph(parse(f)) == """graph f() {
  return @lambda
}

graph lambda(%x) {
  return %x
}
"""


def test_subscript_index():
    def f(x, i):
        return x[i]

    assert str_graph(parse(f)) == """graph f(%x, %i) {
  %_apply0 = <built-in function getitem>(%x, %i)
  return %_apply0
}
"""


def test_unary():
    def f(x):
        return -x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function neg>(%x)
  return %_apply0
}
"""


def test_assign():
    def f():
        x, y = 1, 2
        return y

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = <built-in function getitem>(%_apply0, 0)
  %_apply2 = <built-in function getitem>(%_apply0, 1)
  return %_apply2
}
"""


@pytest.mark.xfail
def test_assign2():
    def f():
        x, *y = 1, 2, 3
        return y

    assert str_graph(parse(f)) == """graph f() {
}
"""


def test_if():
    def f(b, x, y):
        if b:
            return x
        else:
            return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = universe_setitem(%_apply1, %x)
  %_apply2 = universe_setitem(%_apply3, %y)
  %_apply4 = <built-in function truth>(%b)
  %_apply5 = user_switch(%_apply4, @if_true, @if_false)
  %_apply6 = %_apply5()
  return %_apply6
}

graph if_false() {
  %_apply7 = universe_getitem(%_apply3)
  return %_apply7
}

graph if_true() {
  %_apply8 = universe_getitem(%_apply1)
  return %_apply8
}
"""


def test_if2():
    def f(b, x, y):
        if b:
            return x
        return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = universe_setitem(%_apply1, %x)
  %_apply2 = universe_setitem(%_apply3, %y)
  %_apply4 = <built-in function truth>(%b)
  %_apply5 = user_switch(%_apply4, @if_true, @if_false)
  %_apply6 = %_apply5()
  return %_apply6
}

graph if_false() {
  %_apply7 = @if_after()
  return %_apply7
}

graph if_after() {
  %_apply8 = universe_getitem(%_apply3)
  return %_apply8
}

graph if_true() {
  %_apply9 = universe_getitem(%_apply1)
  return %_apply9
}
"""


def test_while():
    def f(b, x, y):
        while b:
            return x
        return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = universe_setitem(%_apply1, %b)
  %_apply2 = universe_setitem(%_apply3, %x)
  %_apply4 = universe_setitem(%_apply5, %y)
  %_apply6 = @while_header()
  return %_apply6
}

graph while_header() {
  %_apply7 = universe_getitem(%_apply1)
  %_apply8 = user_switch(%_apply7, @while_body, @while_else)
  %_apply9 = %_apply8()
  return %_apply9
}

graph while_else() {
  %_apply10 = @while_after()
  return %_apply10
}

graph while_after() {
  %_apply11 = universe_getitem(%_apply5)
  return %_apply11
}

graph while_body() {
  %_apply12 = universe_getitem(%_apply3)
  return %_apply12
}
"""


def test_while2():
    def f(x):
        while x:
            x = x - 1
        return x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = universe_setitem(%_apply1, %x)
  %_apply2 = @while_header()
  return %_apply2
}

graph while_header() {
  %_apply3 = universe_getitem(%_apply1)
  %_apply4 = user_switch(%_apply3, @while_body, @while_else)
  %_apply5 = %_apply4()
  return %_apply5
}

graph while_else() {
  %_apply6 = @while_after()
  return %_apply6
}

graph while_after() {
  %_apply7 = universe_getitem(%_apply1)
  return %_apply7
}

graph while_body() {
  %_apply8 = universe_getitem(%_apply1)
  %_apply9 = <built-in function sub>(%_apply8, 1)
  %_apply10 = universe_setitem(%_apply1, %_apply9)
  %_apply11 = @while_header()
  return %_apply11
}
"""
