import pytest

from myia.parser import parse, MyiaSyntaxError
from myia.ir.print import str_graph
from myia.utils.info import enable_debug


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

    f._myia_flags = {"name": "f22", "test_flag": "inner"}

    assert (
        str_graph(parse(f))
        == """graph @f22() {
  return @g
}

graph @g() {
  return 0
}
"""
    )


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

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  return %x
}
"""
    )


def test_free():
    def f():  # pragma: nocover
        return x

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""
    )


def test_global():
    def f():  # pragma: nocover
        global x
        return x

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = resolve(:tests.test_parser, x)
  return %_apply0
}
"""
    )


def test_nonlocal():
    def f():  # pragma: nocover
        x = 1

        def g():
            nonlocal x
            x = x + 1

        g()
        return x

    assert (
        str_graph(parse(f), allow_cycles=True)
        == """graph @f() {
  %_apply0 = typeof(1)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, 1)
  %_apply3 = @g()
  %_apply4 = universe_getitem(%_apply1)
  return %_apply4
}

graph @g() {
  %_apply5 = universe_getitem(%_apply1)
  %_apply6 = <built-in function add>(%_apply5, 1)
  %_apply7 = universe_setitem(%_apply1, %_apply6)
  return None
}
"""
    )


def test_entry_defaults():
    def f(x=0):  # pragma: nocover
        return x

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_seq():
    def f(x):  # pragma: nocover
        x = x + 1
        return 0

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = <built-in function add>(%x, 1)
  return 0
}
"""
    )


def test_seq2():
    def f(x):  # pragma: nocover
        return x + x

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = <built-in function add>(%x, %x)
  return %_apply0
}
"""
    )


def test_def():
    def f():  # pragma: nocover
        def g(a):
            return a

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a) {
  return %a
}
"""
    )


def test_def2():
    def f():  # pragma: nocover
        def g(a, *b):
            return a

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_def3():
    def f():  # pragma: nocover
        def g(a, b=1):
            return b

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a, %b) {
  return %b
}
"""
    )


def test_def4():
    def f():  # pragma: nocover
        def g(a, **b):
            return a

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_def6():
    def f():  # pragma: nocover
        def g(a: int) -> int:
            return a

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a) {
  return %a
}
"""
    )


def test_def5():
    def f():  # pragma: nocover
        def g(a, *, b):
            return b

        return g

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @g
}

graph @g(%a, %b) {
  return %b
}
"""
    )


def test_getattr():
    def f(a):  # pragma: nocover
        return a.b

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = <built-in function getattr>(%a, b)
  return %_apply0
}
"""
    )


def test_binop():
    def f(a, b):  # pragma: nocover
        return a / b

    assert (
        str_graph(parse(f))
        == """graph @f(%a, %b) {
  %_apply0 = <built-in function truediv>(%a, %b)
  return %_apply0
}
"""
    )


def test_binop2():
    def f(x, y):  # pragma: nocover
        return x + y

    assert (
        str_graph(parse(f))
        == """graph @f(%x, %y) {
  %_apply0 = <built-in function add>(%x, %y)
  return %_apply0
}
"""
    )


def test_binop3():
    def f(x, y):  # pragma: nocover
        return x not in y

    assert (
        str_graph(parse(f))
        == """graph @f(%x, %y) {
  %_apply0 = <built-in function contains>(%x, %y)
  %_apply1 = <built-in function not_>(%_apply0)
  return %_apply1
}
"""
    )


def test_boolop():
    def f(a, b, c):  # pragma: nocover
        return a and b or c

    assert (
        str_graph(parse(f))
        == """graph @f(%a, %b, %c) {
  %_apply0 = typeof(%b)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %b)
  %_apply3 = typeof(%c)
  %_apply4 = make_handle(%_apply3)
  %_apply5 = universe_setitem(%_apply4, %c)
  %_apply6 = <built-in function truth>(%a)
  %_apply7 = switch(%_apply6, @if_true, @if_false)
  %_apply8 = %_apply7()
  %_apply9 = <built-in function truth>(%_apply8)
  %_apply10 = switch(%_apply9, @if_true1, @if_false1)
  %_apply11 = %_apply10()
  return %_apply11
}

graph @if_false() {
  return False
}

graph @if_true() {
  %_apply12 = universe_getitem(%_apply1)
  return %_apply12
}

graph @if_false1() {
  %_apply13 = universe_getitem(%_apply4)
  return %_apply13
}

graph @if_true1() {
  return True
}
"""
    )


def test_call():
    def f():  # pragma: nocover
        def g(a):
            return a

        return g(1)

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = @g(1)
  return %_apply0
}

graph @g(%a) {
  return %a
}
"""
    )


def test_call2():
    def f():  # pragma: nocover
        def g(a, b):
            return a

        return g(1, b=2)

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_dict(b, 2)
  %_apply1 = make_tuple(1)
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call3():
    def f():  # pragma: nocover
        def g(a, b=2):
            return a

        return g(1)

    # XXX: This is probably wrong
    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = @g(1)
  return %_apply0
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call4():
    def f():  # pragma: nocover
        def g(a, b=2):
            return a

        return g(1, 2)

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = @g(1, 2)
  return %_apply0
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call5():
    def f():  # pragma: nocover
        def g(a, *b):
            return a

        return g(1, 2, 3)

    # XXX: This is probably wrong
    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = @g(1, 2, 3)
  return %_apply0
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call6():
    def f():  # pragma: nocover
        def g(a, **b):
            return a

        return g(1, b=2, c=3)

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_dict(b, 2, c, 3)
  %_apply1 = make_tuple(1)
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call7():
    def f():  # pragma: nocover
        def g(a, b):
            return a

        return g(*(1, 2))

    # XXX: This doesn't seem right
    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = make_tuple()
  %_apply2 = apply(@g, %_apply1, %_apply0)
  return %_apply2
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call8():
    def f():  # pragma: nocover
        def g(*, a, b):
            return a

        return g(**{"a": 1, "b": 2})

    # XXX: This doesn't seem right
    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_dict(a, 1, b, 2)
  %_apply1 = make_dict()
  %_apply2 = make_tuple()
  %_apply3 = apply(@g, %_apply2, %_apply0, %_apply1)
  return %_apply3
}

graph @g(%a, %b) {
  return %a
}
"""
    )


def test_call_order():
    def f(a, b, c, d, e, f):  # pragma: nocover
        def g(*a, **b):
            return a

        return g(a + b, c + d, c=33, e=e + f)

    assert (
        str_graph(parse(f))
        == """graph @f(%a, %b, %c, %d, %e, %f) {
  %_apply0 = <built-in function add>(%a, %b)
  %_apply1 = <built-in function add>(%c, %d)
  %_apply2 = <built-in function add>(%e, %f)
  %_apply3 = make_dict(c, 33, e, %_apply2)
  %_apply4 = make_tuple(%_apply0, %_apply1)
  %_apply5 = apply(@g, %_apply4, %_apply3)
  return %_apply5
}

graph @g(%a1, %b1) {
  return %a1
}
"""
    )


def test_compare():
    def f(x):  # pragma: nocover
        return x > 0

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = <built-in function gt>(%x, 0)
  return %_apply0
}
"""
    )


def test_compare2():
    def f(x):  # pragma: nocover
        return 0 < x < 42

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = typeof(%x)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %x)
  %_apply3 = universe_getitem(%_apply1)
  %_apply4 = <built-in function lt>(0, %_apply3)
  %_apply5 = <built-in function truth>(%_apply4)
  %_apply6 = switch(%_apply5, @if_true, @if_false)
  %_apply7 = %_apply6()
  return %_apply7
}

graph @if_false() {
  return False
}

graph @if_true() {
  %_apply8 = universe_getitem(%_apply1)
  %_apply9 = <built-in function lt>(%_apply8, 42)
  return %_apply9
}
"""
    )


def test_dict():
    def f():  # pragma: nocover
        return {"a": 1}

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_dict(a, 1)
  return %_apply0
}
"""
    )


def test_dict2():
    def f():  # pragma: nocover
        a = 1
        b = 2
        c = 3
        return {a + b: c - b, a * c: b / c}

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = <built-in function add>(1, 2)
  %_apply1 = <built-in function sub>(3, 2)
  %_apply2 = <built-in function mul>(1, 3)
  %_apply3 = <built-in function truediv>(2, 3)
  %_apply4 = make_dict(%_apply0, %_apply1, %_apply2, %_apply3)
  return %_apply4
}
"""
    )


def test_extslice():
    def f(a):  # pragma: nocover
        return a[1:2, 3]

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = slice(1, 2, None)
  %_apply1 = make_tuple(%_apply0, 3)
  %_apply2 = <built-in function getitem>(%a, %_apply1)
  return %_apply2
}
"""
    )


def test_ifexp():
    def f(x, y, b):  # pragma: nocover
        return x if b else y

    assert (
        str_graph(parse(f))
        == """graph @f(%x, %y, %b) {
  %_apply0 = typeof(%x)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %x)
  %_apply3 = typeof(%y)
  %_apply4 = make_handle(%_apply3)
  %_apply5 = universe_setitem(%_apply4, %y)
  %_apply6 = <built-in function truth>(%b)
  %_apply7 = user_switch(%_apply6, @if_true, @if_false)
  %_apply8 = %_apply7()
  return %_apply8
}

graph @if_false() {
  %_apply9 = universe_getitem(%_apply4)
  return %_apply9
}

graph @if_true() {
  %_apply10 = universe_getitem(%_apply1)
  return %_apply10
}
"""
    )


def test_index():
    def f(a):  # pragma: nocover
        return a[0]

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = <built-in function getitem>(%a, 0)
  return %_apply0
}
"""
    )


def test_lambda():
    def f():  # pragma: nocover
        l = lambda x: x
        return l

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return @lambda
}

graph @lambda(%x) {
  return %x
}
"""
    )


def test_list():
    def f(a, b):  # pragma: nocover
        c = 4
        return [a + b, c - a, 0, c - b]

    assert (
        str_graph(parse(f))
        == """graph @f(%a, %b) {
  %_apply0 = <built-in function add>(%a, %b)
  %_apply1 = <built-in function sub>(4, %a)
  %_apply2 = <built-in function sub>(4, %b)
  %_apply3 = make_list(%_apply0, %_apply1, 0, %_apply2)
  return %_apply3
}
"""
    )


def test_named_constant():
    def f():  # pragma: nocover
        return True

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return True
}
"""
    )


def test_slice():
    def f(a):  # pragma: nocover
        return a[1::2, :1]

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = slice(1, None, 2)
  %_apply1 = slice(None, 1, None)
  %_apply2 = make_tuple(%_apply0, %_apply1)
  %_apply3 = <built-in function getitem>(%a, %_apply2)
  return %_apply3
}
"""
    )


def test_empty_tuple():
    def f():  # pragma: nocover
        return ()

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return ()
}
"""
    )


def test_unary():
    def f(x):  # pragma: nocover
        return -x

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = <built-in function neg>(%x)
  return %_apply0
}
"""
    )


def test_ann_assign():
    def f():  # pragma: nocover
        a: int = 1
        return a

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return 1
}
"""
    )


def test_assert():
    def f(a):  # pragma: nocover
        assert a == 1, "not 1"

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = <built-in function eq>(%a, 1)
  %_apply1 = <built-in function truth>(%_apply0)
  %_apply2 = user_switch(%_apply1, @if_true, @if_false)
  %_apply3 = %_apply2()
  return %_apply3
}

graph @if_false() {
  %_apply4 = exception(not 1)
  %_apply5 = raise(%_apply4)
  return %_apply5
}

graph @if_true() {
  return None
}
"""
    )


def test_assign():
    def f():  # pragma: nocover
        x, y = 1, 2
        return y

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = <built-in function getitem>(%_apply0, 0)
  %_apply2 = <built-in function getitem>(%_apply0, 1)
  return %_apply2
}
"""
    )


def test_assign2():
    def f():  # pragma: nocover
        [x, y] = 1, 2
        return y

    assert (
        str_graph(parse(f))
        == """graph @f() {
  %_apply0 = make_tuple(1, 2)
  %_apply1 = <built-in function getitem>(%_apply0, 0)
  %_apply2 = <built-in function getitem>(%_apply0, 1)
  return %_apply2
}
"""
    )


def test_assign3():
    def f():  # pragma: nocover
        x, (y, z) = 1, (2, 3)
        return x, y, z

    assert (
        str_graph(parse(f))
        == """graph @f() {
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
    )


@pytest.mark.xfail
def test_assign4():
    def f(a):  # pragma: nocover
        a.b = 1
        return a

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
}
"""
    )


@pytest.mark.xfail
def test_assign5():
    def f(a):  # pragma: nocover
        a[0] = 1
        return a

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
}
"""
    )


@pytest.mark.xfail
def test_assign6():
    def f():  # pragma: nocover
        x, *y = 1, 2, 3
        return y

    assert (
        str_graph(parse(f))
        == """graph @f() {
}
"""
    )


def test_break_continue():
    def f(a):  # pragma: nocover
        for b in a:
            if b < 2:
                break
            if b > 4:
                continue
        return 0

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = python_iter(%a)
  %_apply1 = @for_header(%_apply0)
  return %_apply1
}

graph @for_header(%it) {
  %_apply2 = python_hasnext(%it)
  %_apply3 = user_switch(%_apply2, @for_body, @for_else)
  %_apply4 = %_apply3()
  return %_apply4
}

graph @for_else() {
  %_apply5 = @for_after()
  return %_apply5
}

graph @for_after() {
  return 0
}

graph @for_body() {
  %_apply6 = python_next(%it)
  %_apply7 = <built-in function getitem>(%_apply6, 0)
  %_apply8 = typeof(%_apply7)
  %_apply9 = make_handle(%_apply8)
  %_apply10 = universe_setitem(%_apply9, %_apply7)
  %_apply11 = <built-in function getitem>(%_apply6, 1)
  %_apply12 = universe_getitem(%_apply9)
  %_apply13 = <built-in function lt>(%_apply12, 2)
  %_apply14 = <built-in function truth>(%_apply13)
  %_apply15 = user_switch(%_apply14, @if_true, @if_false)
  %_apply16 = %_apply15()
  return %_apply16
}

graph @if_false() {
  %_apply17 = @if_after()
  return %_apply17
}

graph @if_after() {
  %_apply18 = universe_getitem(%_apply9)
  %_apply19 = <built-in function gt>(%_apply18, 4)
  %_apply20 = <built-in function truth>(%_apply19)
  %_apply21 = user_switch(%_apply20, @if_true1, @if_false1)
  %_apply22 = %_apply21()
  return %_apply22
}

graph @if_false1() {
  %_apply23 = @if_after1()
  return %_apply23
}

graph @if_after1() {
  %_apply24 = @for_header(%_apply11)
  return %_apply24
}

graph @if_true1() {
  %_apply25 = @for_header(%_apply11)
  return %_apply25
}

graph @if_true() {
  %_apply26 = @for_after()
  return %_apply26
}
"""
    )


def test_for():
    def f(b):  # pragma: nocover
        x = 0
        for a in b:
            x = x + 1
        return x

    assert (
        str_graph(parse(f))
        == """graph @f(%b) {
  %_apply0 = typeof(0)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, 0)
  %_apply3 = python_iter(%b)
  %_apply4 = @for_header(%_apply3)
  return %_apply4
}

graph @for_header(%it) {
  %_apply5 = python_hasnext(%it)
  %_apply6 = user_switch(%_apply5, @for_body, @for_else)
  %_apply7 = %_apply6()
  return %_apply7
}

graph @for_else() {
  %_apply8 = @for_after()
  return %_apply8
}

graph @for_after() {
  %_apply9 = universe_getitem(%_apply1)
  return %_apply9
}

graph @for_body() {
  %_apply10 = python_next(%it)
  %_apply11 = <built-in function getitem>(%_apply10, 0)
  %_apply12 = <built-in function getitem>(%_apply10, 1)
  %_apply13 = universe_getitem(%_apply1)
  %_apply14 = <built-in function add>(%_apply13, 1)
  %_apply15 = universe_setitem(%_apply1, %_apply14)
  %_apply16 = @for_header(%_apply12)
  return %_apply16
}
"""
    )


def test_for2():
    def f(a):  # pragma: nocover
        x = 0
        for b, c in a:
            x = x + 1
        else:
            x = x - 1
        return x

    assert (
        str_graph(parse(f))
        == """graph @f(%a) {
  %_apply0 = typeof(0)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, 0)
  %_apply3 = python_iter(%a)
  %_apply4 = @for_header(%_apply3)
  return %_apply4
}

graph @for_header(%it) {
  %_apply5 = python_hasnext(%it)
  %_apply6 = user_switch(%_apply5, @for_body, @for_else)
  %_apply7 = %_apply6()
  return %_apply7
}

graph @for_else() {
  %_apply8 = universe_getitem(%_apply1)
  %_apply9 = <built-in function sub>(%_apply8, 1)
  %_apply10 = universe_setitem(%_apply1, %_apply9)
  %_apply11 = @for_after()
  return %_apply11
}

graph @for_after() {
  %_apply12 = universe_getitem(%_apply1)
  return %_apply12
}

graph @for_body() {
  %_apply13 = python_next(%it)
  %_apply14 = <built-in function getitem>(%_apply13, 0)
  %_apply15 = <built-in function getitem>(%_apply14, 0)
  %_apply16 = <built-in function getitem>(%_apply14, 1)
  %_apply17 = <built-in function getitem>(%_apply13, 1)
  %_apply18 = universe_getitem(%_apply1)
  %_apply19 = <built-in function add>(%_apply18, 1)
  %_apply20 = universe_setitem(%_apply1, %_apply19)
  %_apply21 = @for_header(%_apply17)
  return %_apply21
}
"""
    )


def test_for3():
    def f(n):
        acc = 0
        for i in range(n):
            for j in range(n):
                acc = acc + j
        return acc

    assert (
        str_graph(parse(f))
        == """graph @f(%n) {
  %_apply0 = typeof(%n)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %n)
  %_apply3 = typeof(0)
  %_apply4 = make_handle(%_apply3)
  %_apply5 = universe_setitem(%_apply4, 0)
  %_apply6 = resolve(:tests.test_parser, range)
  %_apply7 = universe_getitem(%_apply1)
  %_apply8 = %_apply6(%_apply7)
  %_apply9 = python_iter(%_apply8)
  %_apply10 = @for_header(%_apply9)
  return %_apply10
}

graph @for_header(%it) {
  %_apply11 = python_hasnext(%it)
  %_apply12 = user_switch(%_apply11, @for_body, @for_else)
  %_apply13 = %_apply12()
  return %_apply13
}

graph @for_else() {
  %_apply14 = @for_after()
  return %_apply14
}

graph @for_after() {
  %_apply15 = universe_getitem(%_apply4)
  return %_apply15
}

graph @for_body() {
  %_apply16 = python_next(%it)
  %_apply17 = <built-in function getitem>(%_apply16, 0)
  %_apply18 = <built-in function getitem>(%_apply16, 1)
  %_apply19 = resolve(:tests.test_parser, range)
  %_apply20 = universe_getitem(%_apply1)
  %_apply21 = %_apply19(%_apply20)
  %_apply22 = python_iter(%_apply21)
  %_apply23 = @for_header1(%_apply22)
  return %_apply23
}

graph @for_header1(%it1) {
  %_apply24 = python_hasnext(%it1)
  %_apply25 = user_switch(%_apply24, @for_body1, @for_else1)
  %_apply26 = %_apply25()
  return %_apply26
}

graph @for_else1() {
  %_apply27 = @for_after1()
  return %_apply27
}

graph @for_after1() {
  %_apply28 = @for_header(%_apply18)
  return %_apply28
}

graph @for_body1() {
  %_apply29 = python_next(%it1)
  %_apply30 = <built-in function getitem>(%_apply29, 0)
  %_apply31 = <built-in function getitem>(%_apply29, 1)
  %_apply32 = universe_getitem(%_apply4)
  %_apply33 = <built-in function add>(%_apply32, %_apply30)
  %_apply34 = universe_setitem(%_apply4, %_apply33)
  %_apply35 = @for_header1(%_apply31)
  return %_apply35
}
"""
    )


def test_if():
    def f(b, x, y):  # pragma: nocover
        if b:
            return x
        else:
            return y

    assert (
        str_graph(parse(f))
        == """graph @f(%b, %x, %y) {
  %_apply0 = typeof(%x)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %x)
  %_apply3 = typeof(%y)
  %_apply4 = make_handle(%_apply3)
  %_apply5 = universe_setitem(%_apply4, %y)
  %_apply6 = <built-in function truth>(%b)
  %_apply7 = user_switch(%_apply6, @if_true, @if_false)
  %_apply8 = %_apply7()
  return %_apply8
}

graph @if_false() {
  %_apply9 = universe_getitem(%_apply4)
  return %_apply9
}

graph @if_true() {
  %_apply10 = universe_getitem(%_apply1)
  return %_apply10
}
"""
    )


def test_if2():
    def f(b, x, y):  # pragma: nocover
        if b:
            y = 0
        return y

    assert (
        str_graph(parse(f))
        == """graph @f(%b, %x, %y) {
  %_apply0 = typeof(%y)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %y)
  %_apply3 = <built-in function truth>(%b)
  %_apply4 = user_switch(%_apply3, @if_true, @if_false)
  %_apply5 = %_apply4()
  return %_apply5
}

graph @if_false() {
  %_apply6 = @if_after()
  return %_apply6
}

graph @if_after() {
  %_apply7 = universe_getitem(%_apply1)
  return %_apply7
}

graph @if_true() {
  %_apply8 = universe_setitem(%_apply1, 0)
  %_apply9 = @if_after()
  return %_apply9
}
"""
    )


def test_pass():
    def f():  # pragma: nocover
        pass

    assert (
        str_graph(parse(f))
        == """graph @f() {
  return None
}
"""
    )


def test_while():
    def f(b, x, y):  # pragma: nocover
        while b:
            return x
        return y

    assert (
        str_graph(parse(f))
        == """graph @f(%b, %x, %y) {
  %_apply0 = typeof(%b)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %b)
  %_apply3 = typeof(%x)
  %_apply4 = make_handle(%_apply3)
  %_apply5 = universe_setitem(%_apply4, %x)
  %_apply6 = typeof(%y)
  %_apply7 = make_handle(%_apply6)
  %_apply8 = universe_setitem(%_apply7, %y)
  %_apply9 = @while_header()
  return %_apply9
}

graph @while_header() {
  %_apply10 = universe_getitem(%_apply1)
  %_apply11 = user_switch(%_apply10, @while_body, @while_else)
  %_apply12 = %_apply11()
  return %_apply12
}

graph @while_else() {
  %_apply13 = @while_after()
  return %_apply13
}

graph @while_after() {
  %_apply14 = universe_getitem(%_apply7)
  return %_apply14
}

graph @while_body() {
  %_apply15 = universe_getitem(%_apply4)
  return %_apply15
}
"""
    )


def test_while2():
    def f(x):  # pragma: nocover
        while x:
            x = x - 1
        return x

    assert (
        str_graph(parse(f))
        == """graph @f(%x) {
  %_apply0 = typeof(%x)
  %_apply1 = make_handle(%_apply0)
  %_apply2 = universe_setitem(%_apply1, %x)
  %_apply3 = @while_header()
  return %_apply3
}

graph @while_header() {
  %_apply4 = universe_getitem(%_apply1)
  %_apply5 = user_switch(%_apply4, @while_body, @while_else)
  %_apply6 = %_apply5()
  return %_apply6
}

graph @while_else() {
  %_apply7 = @while_after()
  return %_apply7
}

graph @while_after() {
  %_apply8 = universe_getitem(%_apply1)
  return %_apply8
}

graph @while_body() {
  %_apply9 = universe_getitem(%_apply1)
  %_apply10 = <built-in function sub>(%_apply9, 1)
  %_apply11 = universe_setitem(%_apply1, %_apply10)
  %_apply12 = @while_header()
  return %_apply12
}
"""
    )


def test_debug():
    def f(a, b, c):  # pragma: no cover
        d = 33

        def g(e, f):
            h = e + f
            i = e - f
            return h - i

        d = d - g(a, b)
        return c - d

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a, b, c) {
  #1 = g(a, b)
  d = <built-in function sub>(33, #1)
  #2 = <built-in function sub>(c, d)
  return #2
}

graph g(e, f.2) {
  h = <built-in function add>(e, f.2)
  i = <built-in function sub>(e, f.2)
  #3 = <built-in function sub>(h, i)
  return #3
}
"""
        )
