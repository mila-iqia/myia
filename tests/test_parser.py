import pytest

from myia.ir.print import str_graph
from myia.parser import MyiaSyntaxError, parse
from myia.utils.info import enable_debug


def test_same():
    def f():  # pragma: no cover
        return 1

    pf1 = parse(f)
    assert pf1 is parse(f)


def test_flags():
    def f():  # pragma: no cover
        def g():
            return 0

        return g

    f._myia_flags = {"name": "f22", "test_flag": "inner"}

    assert (
        str_graph(parse(f))
        == """graph #1() {
  return #2
}

graph #2() {
  return 0
}
"""
    )


def test_var_error1():
    def f(a):  # pragma: no cover
        a = x  # noqa: F841
        x = 1  # noqa: F841

    with pytest.raises(UnboundLocalError):
        parse(f)


def test_var_error2():
    def f():  # pragma: no cover
        global x
        x = 1

    with pytest.raises(NotImplementedError):
        parse(f)


def test_not_supported():
    def f():  # pragma: no cover
        async def g():
            pass

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_simple():
    def f(x):  # pragma: no cover
        return x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  return x
}
"""
        )


def test_free():
    def f():  # pragma: no cover
        return x

    assert (
        str_graph(parse(f))
        == """graph #1() {
  #2 = resolve(:tests.test_parser, x)
  return #2
}
"""
    )


def test_global():
    def f():  # pragma: no cover
        global x
        return x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = resolve(:tests.test_parser, x)
  return #1
}
"""
        )


def test_nonlocal():
    def f():  # pragma: no cover
        x = 1

        def g():
            nonlocal x
            x = x + 1

        g()
        return x

    with enable_debug():
        assert (
            str_graph(parse(f), allow_cycles=True)
            == """graph f() {
  #1 = typeof(1)
  x = make_handle(#1)
  #2 = universe_setitem(x, 1)
  #3 = g()
  #4 = universe_getitem(x)
  return #4
}

graph g() {
  #5 = universe_getitem(x)
  #6 = <built-in function add>(#5, 1)
  #7 = universe_setitem(x, #6)
  return None
}
"""
        )


def test_entry_defaults():
    def f(x=0):  # pragma: no cover
        return x

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_seq():
    def f(x):  # pragma: no cover
        x = x + 1
        return 0

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  x.2 = <built-in function add>(x, 1)
  return 0
}
"""
        )


def test_seq2():
    def f(x):  # pragma: no cover
        return x + x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = <built-in function add>(x, x)
  return #1
}
"""
        )


def test_resolve_read():
    def f():
        global a
        global b
        a2 = a + 1
        b2 = b + 2
        return a2 + b2

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = resolve(:tests.test_parser, a)
  a2 = <built-in function add>(#1, 1)
  #2 = resolve(:tests.test_parser, b)
  b2 = <built-in function add>(#2, 2)
  #3 = <built-in function add>(a2, b2)
  return #3
}
"""
        )


def test_def():
    def f():  # pragma: no cover
        def g(a):
            return a

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a) {
  return a
}
"""
        )


def test_def2():
    def f():  # pragma: no cover
        def g(a, *b):
            return a

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a, b) {
  return a
}
"""
        )


def test_def3():
    def f():  # pragma: no cover
        def g(a, b=1):
            return b

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a, b) {
  return b
}
"""
        )


def test_def4():
    def f():  # pragma: no cover
        def g(a, **b):
            return a

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a, b) {
  return a
}
"""
        )


def test_def6():
    def f():  # pragma: no cover
        def g(a: int) -> int:
            return a

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a) {
  return a
}
"""
        )


def test_def5():
    def f():  # pragma: no cover
        def g(a, *, b):
            return b

        return g

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return g
}

graph g(a, b) {
  return b
}
"""
        )


def test_getattr():
    def f(a):  # pragma: no cover
        return a.b

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = <built-in function getattr>(a, b)
  return #1
}
"""
        )


def test_binop():
    def f(a, b):  # pragma: no cover
        return a / b

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a, b) {
  #1 = <built-in function truediv>(a, b)
  return #1
}
"""
        )


def test_binop2():
    def f(x, y):  # pragma: no cover
        return x + y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x, y) {
  #1 = <built-in function add>(x, y)
  return #1
}
"""
        )


def test_binop3():
    def f(x, y):  # pragma: no cover
        return x not in y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x, y) {
  #1 = <built-in function contains>(x, y)
  #2 = <built-in function not_>(#1)
  return #2
}
"""
        )


def test_boolop():
    def f(a, b, c):  # pragma: no cover
        return a and b or c

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a, b, c) {
  #1 = typeof(b)
  b.2 = make_handle(#1)
  #2 = universe_setitem(b.2, b)
  #3 = typeof(c)
  c.2 = make_handle(#3)
  #4 = universe_setitem(c.2, c)
  #5 = <built-in function truth>(a)
  #6 = switch(#5, if_true:f, if_false:f)
  #7 = #6()
  #8 = <built-in function truth>(#7)
  #9 = switch(#8, if_true:f.2, if_false:f.2)
  #10 = #9()
  return #10
}

graph if_false:f() {
  return False
}

graph if_true:f() {
  #11 = universe_getitem(b.2)
  return #11
}

graph if_false:f.2() {
  #12 = universe_getitem(c.2)
  return #12
}

graph if_true:f.2() {
  return True
}
"""
        )


def test_call():
    def f():  # pragma: no cover
        def g(a):
            return a

        return g(1)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = g(1)
  return #1
}

graph g(a) {
  return a
}
"""
        )


def test_call2():
    def f():  # pragma: no cover
        def g(a, b):
            return a

        return g(1, b=2)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_dict(b, 2)
  #2 = make_tuple(1)
  #3 = apply(g, #2, #1)
  return #3
}

graph g(a, b) {
  return a
}
"""
        )


def test_call3():
    def f():  # pragma: no cover
        def g(a, b=2):
            return a

        return g(1)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = g(1)
  return #1
}

graph g(a, b) {
  return a
}
"""
        )


def test_call4():
    def f():  # pragma: no cover
        def g(a, b=2):
            return a

        return g(1, 2)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = g(1, 2)
  return #1
}

graph g(a, b) {
  return a
}
"""
        )


def test_call5():
    def f():  # pragma: no cover
        def g(a, *b):
            return a

        return g(1, 2, 3)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = g(1, 2, 3)
  return #1
}

graph g(a, b) {
  return a
}
"""
        )


def test_call6():
    def f():  # pragma: no cover
        def g(a, **b):
            return a

        return g(1, b=2, c=3)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_dict(b, 2, c, 3)
  #2 = make_tuple(1)
  #3 = apply(g, #2, #1)
  return #3
}

graph g(a, b) {
  return a
}
"""
        )


def test_call7():
    def f():  # pragma: no cover
        def g(a, b):
            return a

        return g(*(1, 2))

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_tuple(1, 2)
  #2 = make_tuple()
  #3 = apply(g, #2, #1)
  return #3
}

graph g(a, b) {
  return a
}
"""
        )


def test_call8():
    def f():  # pragma: no cover
        def g(*, a, b):
            return a

        return g(**{"a": 1, "b": 2})

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_dict(a, 1, b, 2)
  #2 = make_dict()
  #3 = make_tuple()
  #4 = apply(g, #3, #1, #2)
  return #4
}

graph g(a, b) {
  return a
}
"""
        )


def test_call_order():
    def f(a, b, c, d, e, f):  # pragma: no cover
        def g(*a, **b):
            return a

        return g(a + b, c + d, c=33, e=e + f)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a, b, c, d, e, f.2) {
  #1 = <built-in function add>(a, b)
  #2 = <built-in function add>(c, d)
  #3 = <built-in function add>(e, f.2)
  #4 = make_dict(c, 33, e, #3)
  #5 = make_tuple(#1, #2)
  #6 = apply(g, #5, #4)
  return #6
}

graph g(a.2, b.2) {
  return a.2
}
"""
        )


def test_compare():
    def f(x):  # pragma: no cover
        return x > 0

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = <built-in function gt>(x, 0)
  return #1
}
"""
        )


def test_compare2():
    def f(x):  # pragma: no cover
        return 0 < x < 42

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = typeof(x)
  x.2 = make_handle(#1)
  #2 = universe_setitem(x.2, x)
  #3 = universe_getitem(x.2)
  #4 = <built-in function lt>(0, #3)
  #5 = <built-in function truth>(#4)
  #6 = switch(#5, if_true:f, if_false:f)
  #7 = #6()
  return #7
}

graph if_false:f() {
  return False
}

graph if_true:f() {
  #8 = universe_getitem(x.2)
  #9 = <built-in function lt>(#8, 42)
  return #9
}
"""
        )


def test_dict():
    def f():  # pragma: no cover
        return {"a": 1}

    assert (
        str_graph(parse(f))
        == """graph #1() {
  #2 = make_dict(a, 1)
  return #2
}
"""
    )


def test_dict2():
    def f():  # pragma: no cover
        a = 1
        b = 2
        c = 3
        return {a + b: c - b, a * c: b / c}

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = <built-in function add>(1, 2)
  #2 = <built-in function sub>(3, 2)
  #3 = <built-in function mul>(1, 3)
  #4 = <built-in function truediv>(2, 3)
  #5 = make_dict(#1, #2, #3, #4)
  return #5
}
"""
        )


def test_extslice():
    def f(a):  # pragma: no cover
        return a[1:2, 3]

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = slice(1, 2, None)
  #2 = make_tuple(#1, 3)
  #3 = <built-in function getitem>(a, #2)
  return #3
}
"""
        )


def test_ifexp():
    def f(x, y, b):  # pragma: no cover
        return x if b else y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x, y, b) {
  #1 = typeof(x)
  x.2 = make_handle(#1)
  #2 = universe_setitem(x.2, x)
  #3 = typeof(y)
  y.2 = make_handle(#3)
  #4 = universe_setitem(y.2, y)
  #5 = <built-in function truth>(b)
  #6 = user_switch(#5, if_true:f, if_false:f)
  #7 = #6()
  return #7
}

graph if_false:f() {
  #8 = universe_getitem(y.2)
  return #8
}

graph if_true:f() {
  #9 = universe_getitem(x.2)
  return #9
}
"""
        )


def test_index():
    def f(a):  # pragma: no cover
        return a[0]

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = <built-in function getitem>(a, 0)
  return #1
}
"""
        )


def test_lambda():
    def f():  # pragma: no cover
        lm = lambda x: x  # noqa: E731
        return lm

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  return lambda
}

graph lambda(x) {
  return x
}
"""
        )


def test_list():
    def f(a, b):  # pragma: no cover
        c = 4
        return [a + b, c - a, 0, c - b]

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a, b) {
  #1 = <built-in function add>(a, b)
  #2 = <built-in function sub>(4, a)
  #3 = <built-in function sub>(4, b)
  #4 = make_list(#1, #2, 0, #3)
  return #4
}
"""
        )


def test_named_constant():
    def f():  # pragma: no cover
        return True

    assert (
        str_graph(parse(f))
        == """graph #1() {
  return True
}
"""
    )


def test_slice():
    def f(a):  # pragma: no cover
        return a[1::2, :1]

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = slice(1, None, 2)
  #2 = slice(None, 1, None)
  #3 = make_tuple(#1, #2)
  #4 = <built-in function getitem>(a, #3)
  return #4
}
"""
        )


def test_empty_tuple():
    def f():  # pragma: no cover
        return ()

    assert (
        str_graph(parse(f))
        == """graph #1() {
  return ()
}
"""
    )


def test_unary():
    def f(x):  # pragma: no cover
        return -x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = <built-in function neg>(x)
  return #1
}
"""
        )


def test_ann_assign():
    def f():  # pragma: no cover
        a: int = 1
        return a

    assert (
        str_graph(parse(f))
        == """graph #1() {
  return 1
}
"""
    )


def test_assert():
    def f(a):  # pragma: no cover
        assert a == 1, "not 1"

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = <built-in function eq>(a, 1)
  #2 = <built-in function truth>(#1)
  #3 = user_switch(#2, if_true:f, if_false:f)
  #4 = #3()
  return #4
}

graph if_false:f() {
  #5 = exception(not 1)
  #6 = raise(#5)
  return #6
}

graph if_true:f() {
  return None
}
"""
        )


def test_assign():
    def f():  # pragma: no cover
        x, y = 1, 2  # noqa: F841
        return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_tuple(1, 2)
  x = <built-in function getitem>(#1, 0)
  y = <built-in function getitem>(#1, 1)
  return y
}
"""
        )


def test_assign2():
    def f():  # pragma: no cover
        [x, y] = 1, 2  # noqa: F841
        return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_tuple(1, 2)
  x = <built-in function getitem>(#1, 0)
  y = <built-in function getitem>(#1, 1)
  return y
}
"""
        )


def test_assign3():
    def f():  # pragma: no cover
        x, (y, z) = 1, (2, 3)
        return x, y, z

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = make_tuple(2, 3)
  #2 = make_tuple(1, #1)
  x = <built-in function getitem>(#2, 0)
  #3 = <built-in function getitem>(#2, 1)
  y = <built-in function getitem>(#3, 0)
  z = <built-in function getitem>(#3, 1)
  #4 = make_tuple(x, y, z)
  return #4
}
"""
        )


@pytest.mark.xfail
def test_assign4():
    def f(a):  # pragma: no cover
        a.b = 1
        return a

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
}
"""
        )


@pytest.mark.xfail
def test_assign5():
    def f(a):  # pragma: no cover
        a[0] = 1
        return a

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
}
"""
        )


@pytest.mark.xfail
def test_assign6():
    def f():  # pragma: no cover
        x, *y = 1, 2, 3  # noqa: F841
        return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
}
"""
        )


def test_break_continue():
    def f(a):  # pragma: no cover
        for b in a:
            if b < 2:
                break
            if b > 4:
                continue
        return 0

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = python_iter(a)
  #2 = for_header(#1)
  return #2
}

graph for_header(it) {
  #3 = python_hasnext(it)
  #4 = user_switch(#3, for_body, for_else)
  #5 = #4()
  return #5
}

graph for_else() {
  #6 = for_after()
  return #6
}

graph for_after() {
  return 0
}

graph for_body() {
  #7 = python_next(it)
  #8 = <built-in function getitem>(#7, 0)
  #9 = typeof(#8)
  b = make_handle(#9)
  #10 = universe_setitem(b, #8)
  #11 = <built-in function getitem>(#7, 1)
  #12 = universe_getitem(b)
  #13 = <built-in function lt>(#12, 2)
  #14 = <built-in function truth>(#13)
  #15 = user_switch(#14, if_true:for_body, if_false:for_body)
  #16 = #15()
  return #16
}

graph if_false:for_body() {
  #17 = if_after()
  return #17
}

graph if_after() {
  #18 = universe_getitem(b)
  #19 = <built-in function gt>(#18, 4)
  #20 = <built-in function truth>(#19)
  #21 = user_switch(#20, if_true:if_after, if_false:if_after)
  #22 = #21()
  return #22
}

graph if_false:if_after() {
  #23 = if_after.2()
  return #23
}

graph if_after.2() {
  #24 = for_header(#11)
  return #24
}

graph if_true:if_after() {
  #25 = for_header(#11)
  return #25
}

graph if_true:for_body() {
  #26 = for_after()
  return #26
}
"""
        )


def test_for():
    def f(b):  # pragma: no cover
        x = 0
        for a in b:
            x = x + 1
        return x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b) {
  #1 = typeof(0)
  x = make_handle(#1)
  #2 = universe_setitem(x, 0)
  #3 = python_iter(b)
  #4 = for_header(#3)
  return #4
}

graph for_header(it) {
  #5 = python_hasnext(it)
  #6 = user_switch(#5, for_body, for_else)
  #7 = #6()
  return #7
}

graph for_else() {
  #8 = for_after()
  return #8
}

graph for_after() {
  #9 = universe_getitem(x)
  return #9
}

graph for_body() {
  #10 = python_next(it)
  a = <built-in function getitem>(#10, 0)
  #11 = <built-in function getitem>(#10, 1)
  #12 = universe_getitem(x)
  #13 = <built-in function add>(#12, 1)
  #14 = universe_setitem(x, #13)
  #15 = for_header(#11)
  return #15
}
"""
        )


def test_for2():
    def f(a):  # pragma: no cover
        x = 0
        for b, c in a:
            x = x + 1
        else:
            x = x - 1
        return x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = typeof(0)
  x = make_handle(#1)
  #2 = universe_setitem(x, 0)
  #3 = python_iter(a)
  #4 = for_header(#3)
  return #4
}

graph for_header(it) {
  #5 = python_hasnext(it)
  #6 = user_switch(#5, for_body, for_else)
  #7 = #6()
  return #7
}

graph for_else() {
  #8 = universe_getitem(x)
  #9 = <built-in function sub>(#8, 1)
  #10 = universe_setitem(x, #9)
  #11 = for_after()
  return #11
}

graph for_after() {
  #12 = universe_getitem(x)
  return #12
}

graph for_body() {
  #13 = python_next(it)
  #14 = <built-in function getitem>(#13, 0)
  b = <built-in function getitem>(#14, 0)
  c = <built-in function getitem>(#14, 1)
  #15 = <built-in function getitem>(#13, 1)
  #16 = universe_getitem(x)
  #17 = <built-in function add>(#16, 1)
  #18 = universe_setitem(x, #17)
  #19 = for_header(#15)
  return #19
}
"""
        )


def test_for3():
    def f(n):  # pragma: no cover
        acc = 0
        for i in range(n):
            for j in range(n):
                acc = acc + j
        return acc

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(n) {
  #1 = typeof(n)
  n.2 = make_handle(#1)
  #2 = universe_setitem(n.2, n)
  #3 = typeof(0)
  acc = make_handle(#3)
  #4 = universe_setitem(acc, 0)
  #5 = resolve(:tests.test_parser, range)
  #6 = universe_getitem(n.2)
  #7 = #5(#6)
  #8 = python_iter(#7)
  #9 = for_header(#8)
  return #9
}

graph for_header(it) {
  #10 = python_hasnext(it)
  #11 = user_switch(#10, for_body, for_else)
  #12 = #11()
  return #12
}

graph for_else() {
  #13 = for_after()
  return #13
}

graph for_after() {
  #14 = universe_getitem(acc)
  return #14
}

graph for_body() {
  #15 = python_next(it)
  i = <built-in function getitem>(#15, 0)
  #16 = <built-in function getitem>(#15, 1)
  #17 = resolve(:tests.test_parser, range)
  #18 = universe_getitem(n.2)
  #19 = #17(#18)
  #20 = python_iter(#19)
  #21 = for_header.2(#20)
  return #21
}

graph for_header.2(it.2) {
  #22 = python_hasnext(it.2)
  #23 = user_switch(#22, for_body.2, for_else.2)
  #24 = #23()
  return #24
}

graph for_else.2() {
  #25 = for_after.2()
  return #25
}

graph for_after.2() {
  #26 = for_header(#16)
  return #26
}

graph for_body.2() {
  #27 = python_next(it.2)
  j = <built-in function getitem>(#27, 0)
  #28 = <built-in function getitem>(#27, 1)
  #29 = universe_getitem(acc)
  #30 = <built-in function add>(#29, j)
  #31 = universe_setitem(acc, #30)
  #32 = for_header.2(#28)
  return #32
}
"""
        )


def test_if():
    def f(b, x, y):  # pragma: no cover
        if b:
            return x
        else:
            return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b, x, y) {
  #1 = typeof(x)
  x.2 = make_handle(#1)
  #2 = universe_setitem(x.2, x)
  #3 = typeof(y)
  y.2 = make_handle(#3)
  #4 = universe_setitem(y.2, y)
  #5 = <built-in function truth>(b)
  #6 = user_switch(#5, if_true:f, if_false:f)
  #7 = #6()
  return #7
}

graph if_false:f() {
  #8 = universe_getitem(y.2)
  return #8
}

graph if_true:f() {
  #9 = universe_getitem(x.2)
  return #9
}
"""
        )


def test_if2():
    def f(b, x, y):  # pragma: no cover
        if b:
            y = 0
        return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b, x, y) {
  #1 = typeof(y)
  y.2 = make_handle(#1)
  #2 = universe_setitem(y.2, y)
  #3 = <built-in function truth>(b)
  #4 = user_switch(#3, if_true:f, if_false:f)
  #5 = #4()
  return #5
}

graph if_false:f() {
  #6 = if_after()
  return #6
}

graph if_after() {
  #7 = universe_getitem(y.2)
  return #7
}

graph if_true:f() {
  #8 = universe_setitem(y.2, 0)
  #9 = if_after()
  return #9
}
"""
        )


def test_pass():
    def f():  # pragma: no cover
        pass

    assert (
        str_graph(parse(f))
        == """graph #1() {
  return None
}
"""
    )


def test_while():
    def f(b, x, y):  # pragma: no cover
        while b:
            return x
        return y

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b, x, y) {
  #1 = typeof(b)
  b.2 = make_handle(#1)
  #2 = universe_setitem(b.2, b)
  #3 = typeof(x)
  x.2 = make_handle(#3)
  #4 = universe_setitem(x.2, x)
  #5 = typeof(y)
  y.2 = make_handle(#5)
  #6 = universe_setitem(y.2, y)
  #7 = while_header()
  return #7
}

graph while_header() {
  #8 = universe_getitem(b.2)
  #9 = user_switch(#8, while_body, while_else)
  #10 = #9()
  return #10
}

graph while_else() {
  #11 = while_after()
  return #11
}

graph while_after() {
  #12 = universe_getitem(y.2)
  return #12
}

graph while_body() {
  #13 = universe_getitem(x.2)
  return #13
}
"""
        )


def test_while2():
    def f(x):  # pragma: no cover
        while x:
            x = x - 1
        return x

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = typeof(x)
  x.2 = make_handle(#1)
  #2 = universe_setitem(x.2, x)
  #3 = while_header()
  return #3
}

graph while_header() {
  #4 = universe_getitem(x.2)
  #5 = user_switch(#4, while_body, while_else)
  #6 = #5()
  return #6
}

graph while_else() {
  #7 = while_after()
  return #7
}

graph while_after() {
  #8 = universe_getitem(x.2)
  return #8
}

graph while_body() {
  #9 = universe_getitem(x.2)
  #10 = <built-in function sub>(#9, 1)
  #11 = universe_setitem(x.2, #10)
  #12 = while_header()
  return #12
}
"""
        )
