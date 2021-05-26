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
  #2 = myia.basics.resolve(:tests.test_parser, x)
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
  #1 = myia.basics.resolve(:tests.test_parser, x)
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
  #1 = type(1)
  x = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x, 1)
  #3 = g()
  #4 = myia.basics.global_universe_getitem(x)
  return #4
}

graph g() {
  #5 = myia.basics.global_universe_getitem(x)
  #6 = _operator.add(#5, 1)
  #7 = myia.basics.global_universe_setitem(x, #6)
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
  x.2 = _operator.add(x, 1)
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
  #1 = _operator.add(x, x)
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
  #1 = myia.basics.resolve(:tests.test_parser, a)
  a2 = _operator.add(#1, 1)
  #2 = myia.basics.resolve(:tests.test_parser, b)
  b2 = _operator.add(#2, 2)
  #3 = _operator.add(a2, b2)
  return #3
}
"""
        )


def test_self_recursion():
    def f():
        def g():
            return g()

        return g()

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = type(g)
  g.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(g.2, g)
  #3 = myia.basics.global_universe_getitem(g.2)
  #4 = #3()
  return #4
}

graph g() {
  #5 = myia.basics.global_universe_getitem(g.2)
  #6 = #5()
  return #6
}
"""
        )


def test_nested_resolve():
    def f(b):
        if b:
            a = 0
        else:
            a = 1
        return a

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b) {
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2()
  return #3
}

graph f:if_false() {
  #4 = type(0)
  a = myia.basics.make_handle(#4)
  #5 = myia.basics.global_universe_setitem(a, 1)
  #6 = f:if_after()
  return #6
}

graph f:if_after() {
  #7 = myia.basics.global_universe_getitem(a)
  return #7
}

graph f:if_true() {
  #8 = myia.basics.global_universe_setitem(a, 0)
  #9 = f:if_after()
  return #9
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
  #1 = getattr(a, b)
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
  #1 = _operator.truediv(a, b)
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
  #1 = _operator.add(x, y)
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
  #1 = _operator.contains(x, y)
  #2 = _operator.not_(#1)
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
  #1 = type(b)
  b.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(b.2, b)
  #3 = type(c)
  c.2 = myia.basics.make_handle(#3)
  #4 = myia.basics.global_universe_setitem(c.2, c)
  #5 = _operator.truth(a)
  #6 = myia.basics.switch(#5, f:if_true, f:if_false)
  #7 = #6()
  #8 = _operator.truth(#7)
  #9 = myia.basics.switch(#8, f:if_true.2, f:if_false.2)
  #10 = #9()
  return #10
}

graph f:if_false() {
  return False
}

graph f:if_true() {
  #11 = myia.basics.global_universe_getitem(b.2)
  return #11
}

graph f:if_false.2() {
  #12 = myia.basics.global_universe_getitem(c.2)
  return #12
}

graph f:if_true.2() {
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
  #1 = myia.basics.make_dict(b, 2)
  #2 = myia.basics.make_tuple(1)
  #3 = myia.basics.apply(g, #2, #1)
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
  #1 = myia.basics.make_dict(b, 2, c, 3)
  #2 = myia.basics.make_tuple(1)
  #3 = myia.basics.apply(g, #2, #1)
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
  #1 = myia.basics.make_tuple(1, 2)
  #2 = myia.basics.make_tuple()
  #3 = myia.basics.apply(g, #2, #1)
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
  #1 = myia.basics.make_dict(a, 1, b, 2)
  #2 = myia.basics.make_dict()
  #3 = myia.basics.make_tuple()
  #4 = myia.basics.apply(g, #3, #1, #2)
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
  #1 = _operator.add(a, b)
  #2 = _operator.add(c, d)
  #3 = _operator.add(e, f.2)
  #4 = myia.basics.make_dict(c, 33, e, #3)
  #5 = myia.basics.make_tuple(#1, #2)
  #6 = myia.basics.apply(g, #5, #4)
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
  #1 = _operator.gt(x, 0)
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
  #1 = type(x)
  x.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x.2, x)
  #3 = myia.basics.global_universe_getitem(x.2)
  #4 = _operator.lt(0, #3)
  #5 = _operator.truth(#4)
  #6 = myia.basics.switch(#5, f:if_true, f:if_false)
  #7 = #6()
  return #7
}

graph f:if_false() {
  return False
}

graph f:if_true() {
  #8 = myia.basics.global_universe_getitem(x.2)
  #9 = _operator.lt(#8, 42)
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
  #2 = myia.basics.make_dict(a, 1)
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
  #1 = _operator.add(1, 2)
  #2 = _operator.sub(3, 2)
  #3 = _operator.mul(1, 3)
  #4 = _operator.truediv(2, 3)
  #5 = myia.basics.make_dict(#1, #2, #3, #4)
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
  #2 = myia.basics.make_tuple(#1, 3)
  #3 = _operator.getitem(a, #2)
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
  #1 = type(x)
  x.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x.2, x)
  #3 = type(y)
  y.2 = myia.basics.make_handle(#3)
  #4 = myia.basics.global_universe_setitem(y.2, y)
  #5 = _operator.truth(b)
  #6 = myia.basics.user_switch(#5, f:if_true, f:if_false)
  #7 = #6()
  return #7
}

graph f:if_false() {
  #8 = myia.basics.global_universe_getitem(y.2)
  return #8
}

graph f:if_true() {
  #9 = myia.basics.global_universe_getitem(x.2)
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
  #1 = _operator.getitem(a, 0)
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
  #1 = _operator.add(a, b)
  #2 = _operator.sub(4, a)
  #3 = _operator.sub(4, b)
  #4 = myia.basics.make_list(#1, #2, 0, #3)
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
  #3 = myia.basics.make_tuple(#1, #2)
  #4 = _operator.getitem(a, #3)
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
  #1 = _operator.neg(x)
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
  #1 = _operator.eq(a, 1)
  #2 = _operator.truth(#1)
  #3 = myia.basics.user_switch(#2, f:if_true, f:if_false)
  #4 = #3()
  return #4
}

graph f:if_false() {
  #5 = Exception(not 1)
  #6 = myia.basics.raise_(#5)
  return #6
}

graph f:if_true() {
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
  #1 = myia.basics.make_tuple(1, 2)
  x = _operator.getitem(#1, 0)
  y = _operator.getitem(#1, 1)
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
  #1 = myia.basics.make_tuple(1, 2)
  x = _operator.getitem(#1, 0)
  y = _operator.getitem(#1, 1)
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
  #1 = myia.basics.make_tuple(2, 3)
  #2 = myia.basics.make_tuple(1, #1)
  x = _operator.getitem(#2, 0)
  #3 = _operator.getitem(#2, 1)
  y = _operator.getitem(#3, 0)
  z = _operator.getitem(#3, 1)
  #4 = myia.basics.make_tuple(x, y, z)
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
  #1 = myia.basics.myia_iter(a)
  #2 = f:for(#1)
  return #2
}

graph f:for(it) {
  #3 = myia.basics.myia_hasnext(it)
  #4 = myia.basics.user_switch(#3, f:for:body, f:for:else)
  #5 = #4()
  return #5
}

graph f:for:else() {
  #6 = f:for_after()
  return #6
}

graph f:for_after() {
  return 0
}

graph f:for:body() {
  #7 = myia.basics.myia_next(it)
  #8 = _operator.getitem(#7, 0)
  #9 = type(#8)
  b = myia.basics.make_handle(#9)
  #10 = myia.basics.global_universe_setitem(b, #8)
  #11 = _operator.getitem(#7, 1)
  #12 = myia.basics.global_universe_getitem(b)
  #13 = _operator.lt(#12, 2)
  #14 = _operator.truth(#13)
  #15 = myia.basics.user_switch(#14, f:for:body:if_true, f:for:body:if_false)
  #16 = #15()
  return #16
}

graph f:for:body:if_false() {
  #17 = f:for:body:if_after()
  return #17
}

graph f:for:body:if_after() {
  #18 = myia.basics.global_universe_getitem(b)
  #19 = _operator.gt(#18, 4)
  #20 = _operator.truth(#19)
  #21 = myia.basics.user_switch(#20, f:for:body:if_after:if_true, f:for:body:if_after:if_false)
  #22 = #21()
  return #22
}

graph f:for:body:if_after:if_false() {
  #23 = f:for:body:if_after:if_after()
  return #23
}

graph f:for:body:if_after:if_after() {
  #24 = f:for(#11)
  return #24
}

graph f:for:body:if_after:if_true() {
  #25 = f:for(#11)
  return #25
}

graph f:for:body:if_true() {
  #26 = f:for_after()
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
  #1 = type(0)
  x = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x, 0)
  #3 = myia.basics.myia_iter(b)
  #4 = f:for(#3)
  return #4
}

graph f:for(it) {
  #5 = myia.basics.myia_hasnext(it)
  #6 = myia.basics.user_switch(#5, f:for:body, f:for:else)
  #7 = #6()
  return #7
}

graph f:for:else() {
  #8 = f:for_after()
  return #8
}

graph f:for_after() {
  #9 = myia.basics.global_universe_getitem(x)
  return #9
}

graph f:for:body() {
  #10 = myia.basics.myia_next(it)
  a = _operator.getitem(#10, 0)
  #11 = _operator.getitem(#10, 1)
  #12 = myia.basics.global_universe_getitem(x)
  #13 = _operator.add(#12, 1)
  #14 = myia.basics.global_universe_setitem(x, #13)
  #15 = f:for(#11)
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
  #1 = type(0)
  x = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x, 0)
  #3 = myia.basics.myia_iter(a)
  #4 = f:for(#3)
  return #4
}

graph f:for(it) {
  #5 = myia.basics.myia_hasnext(it)
  #6 = myia.basics.user_switch(#5, f:for:body, f:for:else)
  #7 = #6()
  return #7
}

graph f:for:else() {
  #8 = myia.basics.global_universe_getitem(x)
  #9 = _operator.sub(#8, 1)
  #10 = myia.basics.global_universe_setitem(x, #9)
  #11 = f:for_after()
  return #11
}

graph f:for_after() {
  #12 = myia.basics.global_universe_getitem(x)
  return #12
}

graph f:for:body() {
  #13 = myia.basics.myia_next(it)
  #14 = _operator.getitem(#13, 0)
  b = _operator.getitem(#14, 0)
  c = _operator.getitem(#14, 1)
  #15 = _operator.getitem(#13, 1)
  #16 = myia.basics.global_universe_getitem(x)
  #17 = _operator.add(#16, 1)
  #18 = myia.basics.global_universe_setitem(x, #17)
  #19 = f:for(#15)
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
  #1 = type(n)
  n.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(n.2, n)
  #3 = type(0)
  acc = myia.basics.make_handle(#3)
  #4 = myia.basics.global_universe_setitem(acc, 0)
  #5 = myia.basics.resolve(:tests.test_parser, range)
  #6 = myia.basics.global_universe_getitem(n.2)
  #7 = #5(#6)
  #8 = myia.basics.myia_iter(#7)
  #9 = f:for(#8)
  return #9
}

graph f:for(it) {
  #10 = myia.basics.myia_hasnext(it)
  #11 = myia.basics.user_switch(#10, f:for:body, f:for:else)
  #12 = #11()
  return #12
}

graph f:for:else() {
  #13 = f:for_after()
  return #13
}

graph f:for_after() {
  #14 = myia.basics.global_universe_getitem(acc)
  return #14
}

graph f:for:body() {
  #15 = myia.basics.myia_next(it)
  i = _operator.getitem(#15, 0)
  #16 = _operator.getitem(#15, 1)
  #17 = myia.basics.resolve(:tests.test_parser, range)
  #18 = myia.basics.global_universe_getitem(n.2)
  #19 = #17(#18)
  #20 = myia.basics.myia_iter(#19)
  #21 = f:for:body:for(#20)
  return #21
}

graph f:for:body:for(it.2) {
  #22 = myia.basics.myia_hasnext(it.2)
  #23 = myia.basics.user_switch(#22, f:for:body:for:body, f:for:body:for:else)
  #24 = #23()
  return #24
}

graph f:for:body:for:else() {
  #25 = f:for:body:for_after()
  return #25
}

graph f:for:body:for_after() {
  #26 = f:for(#16)
  return #26
}

graph f:for:body:for:body() {
  #27 = myia.basics.myia_next(it.2)
  j = _operator.getitem(#27, 0)
  #28 = _operator.getitem(#27, 1)
  #29 = myia.basics.global_universe_getitem(acc)
  #30 = _operator.add(#29, j)
  #31 = myia.basics.global_universe_setitem(acc, #30)
  #32 = f:for:body:for(#28)
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
  #1 = type(x)
  x.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x.2, x)
  #3 = type(y)
  y.2 = myia.basics.make_handle(#3)
  #4 = myia.basics.global_universe_setitem(y.2, y)
  #5 = _operator.truth(b)
  #6 = myia.basics.user_switch(#5, f:if_true, f:if_false)
  #7 = #6()
  return #7
}

graph f:if_false() {
  #8 = myia.basics.global_universe_getitem(y.2)
  return #8
}

graph f:if_true() {
  #9 = myia.basics.global_universe_getitem(x.2)
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
  #1 = type(y)
  y.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(y.2, y)
  #3 = _operator.truth(b)
  #4 = myia.basics.user_switch(#3, f:if_true, f:if_false)
  #5 = #4()
  return #5
}

graph f:if_false() {
  #6 = f:if_after()
  return #6
}

graph f:if_after() {
  #7 = myia.basics.global_universe_getitem(y.2)
  return #7
}

graph f:if_true() {
  #8 = myia.basics.global_universe_setitem(y.2, 0)
  #9 = f:if_after()
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
  #1 = type(b)
  b.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(b.2, b)
  #3 = type(x)
  x.2 = myia.basics.make_handle(#3)
  #4 = myia.basics.global_universe_setitem(x.2, x)
  #5 = type(y)
  y.2 = myia.basics.make_handle(#5)
  #6 = myia.basics.global_universe_setitem(y.2, y)
  #7 = f:while()
  return #7
}

graph f:while() {
  #8 = myia.basics.global_universe_getitem(b.2)
  #9 = myia.basics.user_switch(#8, f:while:body, f:while:else)
  #10 = #9()
  return #10
}

graph f:while:else() {
  #11 = f:while_after()
  return #11
}

graph f:while_after() {
  #12 = myia.basics.global_universe_getitem(y.2)
  return #12
}

graph f:while:body() {
  #13 = myia.basics.global_universe_getitem(x.2)
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
  #1 = type(x)
  x.2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x.2, x)
  #3 = f:while()
  return #3
}

graph f:while() {
  #4 = myia.basics.global_universe_getitem(x.2)
  #5 = myia.basics.user_switch(#4, f:while:body, f:while:else)
  #6 = #5()
  return #6
}

graph f:while:else() {
  #7 = f:while_after()
  return #7
}

graph f:while_after() {
  #8 = myia.basics.global_universe_getitem(x.2)
  return #8
}

graph f:while:body() {
  #9 = myia.basics.global_universe_getitem(x.2)
  #10 = _operator.sub(#9, 1)
  #11 = myia.basics.global_universe_setitem(x.2, #10)
  #12 = f:while()
  return #12
}
"""
        )
