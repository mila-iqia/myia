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


def test_nonlocal():
    def f():
        x = 1
        def g():
            nonlocal x
            x = x + 1
        g()
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = universe_setitem(%_apply1, 1)
  %_apply2 = @g()
  %_apply3 = universe_getitem(%_apply1)
  return %_apply3
}

graph g() {
  %_apply4 = universe_getitem(%_apply1)
  %_apply5 = <built-in function add>(%_apply4, 1)
  %_apply6 = universe_setitem(%_apply1, %_apply5)
  return None
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

def test_def():
    def f():
        def g(a):
            return a
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a) {
  return %a
}
"""

def test_def2():
    def f():
        def g(a, *b):
            return a
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a, %b) {
  return %a
}
"""

def test_def3():
    def f():
        def g(a, b=1):
            return b
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a, %b) {
  return %b
}
"""

def test_def4():
    def f():
        def g(a, **b):
            return a
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a, %b) {
  return %a
}
"""

def test_def5():
    def f():
        def g(a, *, b):
            return b
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a, %b) {
  return %b
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

def test_break():
    def f(a):
        for b in a:
            break
        return 0

    assert  str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = python_iter(%a)
  %_apply1 = @for_header(%_apply0)
  return %_apply1
}

graph for_header(%it) {
  %_apply2 = python_hasnext(%it)
  %_apply3 = user_switch(%_apply2, @for_body, @for_else)
  %_apply4 = %_apply3()
  return %_apply4
}

graph for_else() {
  %_apply5 = @for_after()
  return %_apply5
}

graph for_after() {
  return 0
}

graph for_body() {
  %_apply6 = python_next(%it)
  %_apply7 = <built-in function getitem>(%_apply6, 0)
  %_apply8 = <built-in function getitem>(%_apply6, 1)
  %_apply9 = @for_after()
  return %_apply9
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


def test_for():
    def f(b):
        x = 0
        for a in b:
            x = x + 1
        return x

    assert str_graph(parse(f)) == """graph f(%b) {
  %_apply0 = universe_setitem(%_apply1, 0)
  %_apply2 = python_iter(%b)
  %_apply3 = @for_header(%_apply2)
  return %_apply3
}

graph for_header(%it) {
  %_apply4 = python_hasnext(%it)
  %_apply5 = user_switch(%_apply4, @for_body, @for_else)
  %_apply6 = %_apply5()
  return %_apply6
}

graph for_else() {
  %_apply7 = @for_after()
  return %_apply7
}

graph for_after() {
  %_apply8 = universe_getitem(%_apply1)
  return %_apply8
}

graph for_body() {
  %_apply9 = python_next(%it)
  %_apply10 = <built-in function getitem>(%_apply9, 0)
  %_apply11 = <built-in function getitem>(%_apply9, 1)
  %_apply12 = universe_getitem(%_apply1)
  %_apply13 = <built-in function add>(%_apply12, 1)
  %_apply14 = universe_setitem(%_apply1, %_apply13)
  %_apply15 = @for_header(%_apply11)
  return %_apply15
}
"""


def test_for2():
    def f(a):
        x = 0
        for b, c in a:
            x = x + 1
        else:
            x = x - 1
        return x

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = universe_setitem(%_apply1, 0)
  %_apply2 = python_iter(%a)
  %_apply3 = @for_header(%_apply2)
  return %_apply3
}

graph for_header(%it) {
  %_apply4 = python_hasnext(%it)
  %_apply5 = user_switch(%_apply4, @for_body, @for_else)
  %_apply6 = %_apply5()
  return %_apply6
}

graph for_else() {
  %_apply7 = universe_getitem(%_apply1)
  %_apply8 = <built-in function sub>(%_apply7, 1)
  %_apply9 = universe_setitem(%_apply1, %_apply8)
  %_apply10 = @for_after()
  return %_apply10
}

graph for_after() {
  %_apply11 = universe_getitem(%_apply1)
  return %_apply11
}

graph for_body() {
  %_apply12 = python_next(%it)
  %_apply13 = <built-in function getitem>(%_apply12, 0)
  %_apply14 = <built-in function getitem>(%_apply13, 0)
  %_apply15 = <built-in function getitem>(%_apply13, 1)
  %_apply16 = <built-in function getitem>(%_apply12, 1)
  %_apply17 = universe_getitem(%_apply1)
  %_apply18 = <built-in function add>(%_apply17, 1)
  %_apply19 = universe_setitem(%_apply1, %_apply18)
  %_apply20 = @for_header(%_apply16)
  return %_apply20
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
