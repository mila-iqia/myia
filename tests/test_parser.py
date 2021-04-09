import pytest

from myia.parser import parse, MyiaSyntaxError
from myia.ir.print import str_graph


def test_same():
    def f():  # pragma: nocover
        return 1

    pf1 = parse(f)
    assert pf1 is parse(f)


def test_flags():
    def f():  # pragma: nocover
        def g():
            return 0
        return g

    f._myia_flags = {'name': 'f22', 'test_flag': 'inner'}

    assert str_graph(parse(f)) == """graph f22() {
  return @g
}

graph g() {
  return 0
}
"""


def test_var_error1():
    def f(a):  # pragma: nocover
        a = x
        x = 1

    with pytest.raises(UnboundLocalError):
        parse(f)


def test_var_error2():
    def f():  # pragma: nocover
        global x
        x = 1

    with pytest.raises(NotImplementedError):
        parse(f)


def test_not_supported():
    def f():  # pragma: nocover
        async def g():
            pass

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_simple():
    def f(x):  # pragma: nocover
        return x

    assert str_graph(parse(f)) == """graph f(%x) {
  return %x
}
"""


def test_free():
    def f():  # pragma: nocover
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""


def test_global():
    def f():  # pragma: nocover
        global x
        return x

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""


def test_nonlocal():
    def f():  # pragma: nocover
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


def test_entry_defaults():
    def f(x=0):  # pragma: nocover
        return x

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_seq():
    def f(x):  # pragma: nocover
        x = x + 1
        return 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function add>(%x, 1)
  return 0
}
"""


def test_seq2():
    def f(x):  # pragma: nocover
        return x + x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function add>(%x, %x)
  return %_apply0
}
"""

def test_def():
    def f():  # pragma: nocover
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
    def f():  # pragma: nocover
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
    def f():  # pragma: nocover
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
    def f():  # pragma: nocover
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


def test_def6():
    def f():  # pragma: nocover
        def g(a: int) -> int:
            return a
        return g

    assert str_graph(parse(f)) == """graph f() {
  return @g
}

graph g(%a) {
  return %a
}
"""


def test_def5():
    def f():  # pragma: nocover
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


def test_getattr():
    def f(a):  # pragma: nocover
        return a.b

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = <built-in function getattr>(%a, b)
  return %_apply0
}
"""


def test_binop():
    def f(a, b):  # pragma: nocover
        return a / b

    assert str_graph(parse(f)) == """graph f(%a, %b) {
  %_apply0 = <built-in function truediv>(%a, %b)
  return %_apply0
}
"""


def test_binop2():
    def f(x, y):  # pragma: nocover
        return x + y

    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = <built-in function add>(%x, %y)
  return %_apply0
}
"""


def test_binop3():
    def f(x, y):  # pragma: nocover
        return x not in y

    assert str_graph(parse(f)) == """graph f(%x, %y) {
  %_apply0 = <built-in function contains>(%x, %y)
  %_apply1 = <built-in function not_>(%_apply0)
  return %_apply1
}
"""


def test_boolop():
    def f(a, b, c):  # pragma: nocover
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


def test_call():
    def f():  # pragma: nocover
        def g(a):
            return a

        return g(1)

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = @g(1)
  return %_apply0
}

graph g(%a) {
  return %a
}
"""


def test_call2():
    def f():  # pragma: nocover
        def g(a, b):
            return a

        return g(1, b=2)

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_dict(b, 2)
  %_apply1 = make_tuple(1)
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph g(%a, %b) {
  return %a
}
"""


def test_call3():
    def f():  # pragma: nocover
        def g(a, b=2):
            return a

        return g(1)

    # XXX: This is probably wrong
    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = @g(1)
  return %_apply0
}

graph g(%a, %b) {
  return %a
}
"""


def test_call4():
    def f():  # pragma: nocover
        def g(a, b=2):
            return a

        return g(1, 2)

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = @g(1, 2)
  return %_apply0
}

graph g(%a, %b) {
  return %a
}
"""


def test_call5():
    def f():  # pragma: nocover
        def g(a, *b):
            return a

        return g(1, 2, 3)

    # XXX: This is probably wrong
    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = @g(1, 2, 3)
  return %_apply0
}

graph g(%a, %b) {
  return %a
}
"""


def test_call6():
    def f():  # pragma: nocover
        def g(a, **b):
            return a

        return g(1, b=2, c=3)

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_dict(b, 2, c, 3)
  %_apply1 = make_tuple(1)
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph g(%a, %b) {
  return %a
}
"""


def test_call7():
    def f():  # pragma: nocover
        def g(a, b):
            return a

        return g(*(1, 2))

    # XXX: This doesn't seem right
    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = make_tuple()
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph g(%a, %b) {
  return %a
}
"""


def test_call8():
    def f():  # pragma: nocover
        def g(*, a, b):
            return a

        return g(**{'a': 1, 'b': 2})

    # XXX: This doesn't seem right
    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_dict(a, 1, b, 2)
  %_apply1 = make_dict()
  %_apply2 = make_tuple()
  %_apply3 = apply(@g, %_apply2, %_apply0, %_apply1)
  return %_apply3
}

graph g(%a, %b) {
  return %a
}
"""


def test_call_order():
    def f(a, b, c, d, e, f):  # pragma: nocover
        def g(*a, **b):
            return a

        return g(a+b, c+d, c=33, e=e+f)

    assert str_graph(parse(f)) == """graph f(%a, %b, %c, %d, %e, %f) {
  %_apply0 = <built-in function add>(%a, %b)
  %_apply1 = <built-in function add>(%c, %d)
  %_apply2 = <built-in function add>(%e, %f)
  %_apply3 = make_dict(c, 33, e, %_apply2)
  %_apply4 = make_tuple(%_apply0, %_apply1)
  %_apply5 = apply(@g, %_apply4, %_apply3)
  return %_apply5
}

graph g(%a, %b) {
  return %a
}
"""


def test_compare():
    def f(x):  # pragma: nocover
        return x > 0

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function gt>(%x, 0)
  return %_apply0
}
"""


def test_compare2():
    def f(x):  # pragma: nocover
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


def test_dict():
    def f():  # pragma: nocover
        return {'a': 1}

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_dict(a, 1)
  return %_apply0
}
"""


def test_dict2():
    def f():  # pragma: nocover
        a = 1
        b = 2
        c = 3
        return {a+b: c-b, a*c: b/c}

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = <built-in function add>(1, 2)
  %_apply1 = <built-in function sub>(3, 2)
  %_apply2 = <built-in function mul>(1, 3)
  %_apply3 = <built-in function truediv>(2, 3)
  %_apply4 = make_dict(%_apply0, %_apply1, %_apply2, %_apply3)
  return %_apply4
}
"""


def test_extslice():
    def f(a):  # pragma: nocover
        return a[1:2, 3]

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = slice(1, 2, None)
  %_apply1 = make_tuple(%_apply0, 3)
  %_apply2 = <built-in function getitem>(%a, %_apply1)
  return %_apply2
}
"""


def test_ifexp():
    def f(x, y, b):  # pragma: nocover
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


def test_index():
    def f(a):  # pragma: nocover
        return a[0]

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = <built-in function getitem>(%a, 0)
  return %_apply0
}
"""


def test_lambda():
    def f():  # pragma: nocover
        l = lambda x: x
        return l

    assert str_graph(parse(f)) == """graph f() {
  return @lambda
}

graph lambda(%x) {
  return %x
}
"""


def test_list():
    def f(a, b):  # pragma: nocover
        c = 4
        return [a+b, c-a, 0, c-b]

    assert str_graph(parse(f)) == """graph f(%a, %b) {
  %_apply0 = <built-in function add>(%a, %b)
  %_apply1 = <built-in function sub>(4, %a)
  %_apply2 = <built-in function sub>(4, %b)
  %_apply3 = make_list(%_apply0, %_apply1, 0, %_apply2)
  return %_apply3
}
"""


def test_named_constant():
    def f():  # pragma: nocover
        return True

    assert str_graph(parse(f)) == """graph f() {
  return True
}
"""

def test_slice():
    def f(a):  # pragma: nocover
        return a[1::2, :1]

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = slice(1, None, 2)
  %_apply1 = slice(None, 1, None)
  %_apply2 = make_tuple(%_apply0, %_apply1)
  %_apply3 = <built-in function getitem>(%a, %_apply2)
  return %_apply3
}
"""


def test_empty_tuple():
    def f():  # pragma: nocover
        return ()

    assert str_graph(parse(f)) == """graph f() {
  return ()
}
"""


def test_unary():
    def f(x):  # pragma: nocover
        return -x

    assert str_graph(parse(f)) == """graph f(%x) {
  %_apply0 = <built-in function neg>(%x)
  return %_apply0
}
"""


def test_ann_assign():
    def f():  # pragma: nocover
        a : int = 1
        return a

    assert str_graph(parse(f)) == """graph f() {
  return 1
}
"""


def test_assert():
    def f(a):  # pragma: nocover
        assert a == 1, "not 1"

    assert str_graph(parse(f)) == """graph f(%a) {
  %_apply0 = <built-in function eq>(%a, 1)
  %_apply1 = <built-in function truth>(%_apply0)
  %_apply2 = user_switch(%_apply1, @if_true, @if_false)
  %_apply3 = %_apply2()
  return %_apply3
}

graph if_false() {
  %_apply4 = exception(not 1)
  %_apply5 = raise(%_apply4)
  return %_apply5
}

graph if_true() {
  return None
}
"""


def test_assign():
    def f():  # pragma: nocover
        x, y = 1, 2
        return y

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = <built-in function getitem>(%_apply0, 0)
  %_apply2 = <built-in function getitem>(%_apply0, 1)
  return %_apply2
}
"""


def test_assign2():
    def f():  # pragma: nocover
        [x, y] = 1, 2
        return y

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = <built-in function getitem>(%_apply0, 0)
  %_apply2 = <built-in function getitem>(%_apply0, 1)
  return %_apply2
}
"""


def test_assign3():
    def f():  # pragma: nocover
        x, (y, z) = 1, (2, 3)
        return x, y, z

    assert str_graph(parse(f)) == """graph f() {
  %_apply0 = make_tuple(2, 3)
  %_apply1 = make_tuple(1, %_apply0)
  %_apply2 = <built-in function getitem>(%_apply1, 0)
  %_apply3 = <built-in function getitem>(%_apply1, 1)
  %_apply4 = <built-in function getitem>(%_apply3, 0)
  %_apply5 = <built-in function getitem>(%_apply3, 1)
  %_apply6 = make_tuple(%_apply2, %_apply4, %_apply5)
  return %_apply6
}
"""


@pytest.mark.xfail
def test_assign4():
    def f(a):  # pragma: nocover
        a.b = 1
        return a

    assert str_graph(parse(f)) == """graph f(%a) {
}
"""


@pytest.mark.xfail
def test_assign5():
    def f(a):  # pragma: nocover
        a[0] = 1
        return a

    assert str_graph(parse(f)) == """graph f(%a) {
}
"""


@pytest.mark.xfail
def test_assign6():
    def f():  # pragma: nocover
        x, *y = 1, 2, 3
        return y

    assert str_graph(parse(f)) == """graph f() {
}
"""


def test_break_continue():
    def f(a):  # pragma: nocover
        for b in a:
            if b < 2:
                break
            if b > 4:
                continue
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
  %_apply8 = universe_setitem(%_apply9, %_apply7)
  %_apply10 = <built-in function getitem>(%_apply6, 1)
  %_apply11 = universe_getitem(%_apply9)
  %_apply12 = <built-in function lt>(%_apply11, 2)
  %_apply13 = <built-in function truth>(%_apply12)
  %_apply14 = user_switch(%_apply13, @if_true, @if_false)
  %_apply15 = %_apply14()
  return %_apply15
}

graph if_false() {
  %_apply16 = @if_after()
  return %_apply16
}

graph if_after() {
  %_apply17 = universe_getitem(%_apply9)
  %_apply18 = <built-in function gt>(%_apply17, 4)
  %_apply19 = <built-in function truth>(%_apply18)
  %_apply20 = user_switch(%_apply19, @if_true, @if_false)
  %_apply21 = %_apply20()
  return %_apply21
}

graph if_false() {
  %_apply22 = @if_after()
  return %_apply22
}

graph if_after() {
  %_apply23 = @for_header(%_apply10)
  return %_apply23
}

graph if_true() {
  %_apply24 = @for_header(%_apply10)
  return %_apply24
}

graph if_true() {
  %_apply25 = @for_after()
  return %_apply25
}
"""


def test_for():
    def f(b):  # pragma: nocover
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
    def f(a):  # pragma: nocover
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


def test_if():
    def f(b, x, y):  # pragma: nocover
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
    def f(b, x, y):  # pragma: nocover
        if b:
            y = 0
        return y

    assert str_graph(parse(f)) == """graph f(%b, %x, %y) {
  %_apply0 = universe_setitem(%_apply1, %y)
  %_apply2 = <built-in function truth>(%b)
  %_apply3 = user_switch(%_apply2, @if_true, @if_false)
  %_apply4 = %_apply3()
  return %_apply4
}

graph if_false() {
  %_apply5 = @if_after()
  return %_apply5
}

graph if_after() {
  %_apply6 = universe_getitem(%_apply1)
  return %_apply6
}

graph if_true() {
  %_apply7 = universe_setitem(%_apply1, 0)
  %_apply8 = @if_after()
  return %_apply8
}
"""

def test_pass():
    def f():  # pragma: nocover
        pass

    assert str_graph(parse(f)) == """graph f() {
  return None
}
"""


def test_while():
    def f(b, x, y):  # pragma: nocover
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
    def f(x):  # pragma: nocover
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
