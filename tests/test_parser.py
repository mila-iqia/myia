import pytest

from myia.ir.print import str_graph
from myia.parser import MyiaSyntaxError, parse
from myia.parser_opt import apply_parser_opts
from myia.utils.info import enable_debug

from .common import predictable_placeholders


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
  #2 = myia.basics.resolve(:tests.test_parser, 'x')
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
  #1 = myia.basics.resolve(:tests.test_parser, 'x')
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

    with enable_debug(), predictable_placeholders():
        assert (
            str_graph(parse(f), allow_cycles=True)
            == """graph f() {
  x = myia.basics.make_handle(??1)
  #1 = myia.basics.global_universe_setitem(x, 1)
  #2 = g()
  #3 = myia.basics.global_universe_getitem(x)
  return #3
}

graph g() {
  #4 = myia.basics.global_universe_getitem(x)
  #5 = _operator.add(#4, 1)
  #6 = myia.basics.global_universe_setitem(x, #5)
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
  x~2 = _operator.add(x, 1)
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
    def f():  # pragma: no cover
        global a
        global b
        a2 = a + 1
        b2 = b + 2
        return a2 + b2

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = myia.basics.resolve(:tests.test_parser, 'a')
  a2 = _operator.add(#1, 1)
  #2 = myia.basics.resolve(:tests.test_parser, 'b')
  b2 = _operator.add(#2, 2)
  #3 = _operator.add(a2, b2)
  return #3
}
"""
        )


def test_self_recursion():
    def f():  # pragma: no cover
        def g():
            return g()

        return g()

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f() {
  #1 = type(g)
  g~2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(g~2, g)
  #3 = myia.basics.global_universe_getitem(g~2)
  #4 = #3()
  return #4
}

graph g() {
  #5 = myia.basics.global_universe_getitem(g~2)
  #6 = #5()
  return #6
}
"""
        )


def test_self_recursion_parser_opt():
    def f():  # pragma: no cover
        def g():
            return g()

        return g()

    with enable_debug():
        assert (
            str_graph(apply_parser_opts(parse(f)))
            == """graph f() {
  #1 = g()
  return #1
}

graph g() {
  #2 = g()
  return #2
}
"""
        )


def test_no_return():
    def f(x):  # pragma: no cover
        y = 2 * x

        def g(i):
            j = i + x + y  # noqa: F841

        z = g(0)  # noqa: F841

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(x) {
  #1 = type(x)
  x~2 = myia.basics.make_handle(#1)
  #2 = myia.basics.global_universe_setitem(x~2, x)
  #3 = myia.basics.global_universe_getitem(x~2)
  #4 = _operator.mul(2, #3)
  #5 = type(#4)
  y = myia.basics.make_handle(#5)
  #6 = myia.basics.global_universe_setitem(y, #4)
  z = g(0)
  return None
}

graph g(i) {
  #7 = myia.basics.global_universe_getitem(x~2)
  #8 = _operator.add(i, #7)
  #9 = myia.basics.global_universe_getitem(y)
  j = _operator.add(#8, #9)
  return None
}
"""
        )


def test_no_return_parser_opt():
    def f(x):  # pragma: no cover
        y = 2 * x

        def g(i):
            j = i + x + y  # noqa: F841

        z = g(0)  # noqa: F841

    with enable_debug():
        assert (
            str_graph(apply_parser_opts(parse(f)))
            == """graph f(x) {
  #1 = _operator.mul(2, x)
  z = g(0)
  return None
}

graph g(i) {
  #2 = _operator.add(i, x)
  j = _operator.add(#2, #1)
  return None
}
"""
        )


def test_nested_resolve():
    def f(b):  # pragma: no cover
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
  #4 = f:if_after(1)
  return #4
}

graph f:if_after(phi_a) {
  return phi_a
}

graph f:if_true() {
  #5 = f:if_after(0)
  return #5
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


def test_def_posonly():
    def f(a, b, /, c, d, e):  # pragma: no cover
        pass

    g = parse(f)
    assert len(g.parameters) == 5
    assert g.posonly == 2


def test_getattr():
    def f(a):  # pragma: no cover
        return a.b

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(a) {
  #1 = getattr(a, 'b')
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
  #1 = _operator.truth(a)
  #2 = myia.basics.switch(#1, f:if_true, f:if_false)
  #3 = #2(b)
  #4 = _operator.truth(#3)
  #5 = myia.basics.switch(#4, f:if_true~2, f:if_false~2)
  #6 = #5(c)
  return #6
}

graph f:if_false(phi_b) {
  return False
}

graph f:if_true(phi_b~2) {
  return phi_b~2
}

graph f:if_false~2(phi_c) {
  return phi_c
}

graph f:if_true~2(phi_c~2) {
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
  #1 = myia.basics.make_dict('b', 2)
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
  #1 = myia.basics.make_dict('b', 2, 'c', 3)
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
  #1 = myia.basics.make_dict('a', 1, 'b', 2)
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
            == """graph f(a, b, c, d, e, f~2) {
  #1 = _operator.add(a, b)
  #2 = _operator.add(c, d)
  #3 = _operator.add(e, f~2)
  #4 = myia.basics.make_dict('c', 33, 'e', #3)
  #5 = myia.basics.make_tuple(#1, #2)
  #6 = myia.basics.apply(g, #5, #4)
  return #6
}

graph g(a~2, b~2) {
  return a~2
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
  #1 = _operator.lt(0, x)
  #2 = _operator.truth(#1)
  #3 = myia.basics.switch(#2, f:if_true, f:if_false)
  #4 = #3(x)
  return #4
}

graph f:if_false(phi_x) {
  return False
}

graph f:if_true(phi_x~2) {
  #5 = _operator.lt(phi_x~2, 42)
  return #5
}
"""
        )


def test_dict():
    def f():  # pragma: no cover
        return {"a": 1}

    assert (
        str_graph(parse(f))
        == """graph #1() {
  #2 = myia.basics.make_dict('a', 1)
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
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(x, y)
  return #3
}

graph f:if_false(phi_x, phi_y) {
  return phi_y
}

graph f:if_true(phi_x~2, phi_y~2) {
  return phi_x~2
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
  #5 = Exception('not 1')
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
  b = _operator.getitem(#7, 0)
  #8 = _operator.getitem(#7, 1)
  #9 = _operator.lt(b, 2)
  #10 = _operator.truth(#9)
  #11 = myia.basics.user_switch(#10, f:for:body:if_true, f:for:body:if_false)
  #12 = #11(b)
  return #12
}

graph f:for:body:if_false(phi_b) {
  #13 = f:for:body:if_after(phi_b)
  return #13
}

graph f:for:body:if_after(phi_b~2) {
  #14 = _operator.gt(phi_b~2, 4)
  #15 = _operator.truth(#14)
  #16 = myia.basics.user_switch(#15, f:for:body:if_after:if_true, f:for:body:if_after:if_false)
  #17 = #16()
  return #17
}

graph f:for:body:if_after:if_false() {
  #18 = f:for:body:if_after:if_after()
  return #18
}

graph f:for:body:if_after:if_after() {
  #19 = f:for(#8)
  return #19
}

graph f:for:body:if_after:if_true() {
  #20 = f:for(#8)
  return #20
}

graph f:for:body:if_true(phi_b~3) {
  #21 = f:for_after()
  return #21
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
  #1 = myia.basics.myia_iter(b)
  #2 = f:for(#1, 0)
  return #2
}

graph f:for(it, phi_x) {
  #3 = myia.basics.myia_hasnext(it)
  #4 = myia.basics.user_switch(#3, f:for:body, f:for:else)
  #5 = #4(phi_x)
  return #5
}

graph f:for:else(phi_x~2) {
  #6 = f:for_after(phi_x~2)
  return #6
}

graph f:for_after(phi_x~3) {
  return phi_x~3
}

graph f:for:body(phi_x~4) {
  #7 = myia.basics.myia_next(it)
  a = _operator.getitem(#7, 0)
  #8 = _operator.getitem(#7, 1)
  x = _operator.add(phi_x~4, 1)
  #9 = f:for(#8, x)
  return #9
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
  #1 = myia.basics.myia_iter(a)
  #2 = f:for(#1, 0)
  return #2
}

graph f:for(it, phi_x) {
  #3 = myia.basics.myia_hasnext(it)
  #4 = myia.basics.user_switch(#3, f:for:body, f:for:else)
  #5 = #4(phi_x)
  return #5
}

graph f:for:else(phi_x~2) {
  x = _operator.sub(phi_x~2, 1)
  #6 = f:for_after(x)
  return #6
}

graph f:for_after(phi_x~3) {
  return phi_x~3
}

graph f:for:body(phi_x~4) {
  #7 = myia.basics.myia_next(it)
  #8 = _operator.getitem(#7, 0)
  b = _operator.getitem(#8, 0)
  c = _operator.getitem(#8, 1)
  #9 = _operator.getitem(#7, 1)
  x~2 = _operator.add(phi_x~4, 1)
  #10 = f:for(#9, x~2)
  return #10
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
  #1 = myia.basics.resolve(:tests.test_parser, 'range')
  #2 = #1(n)
  #3 = myia.basics.myia_iter(#2)
  #4 = f:for(#3, n, 0)
  return #4
}

graph f:for(it, phi_n, phi_acc) {
  #5 = myia.basics.myia_hasnext(it)
  #6 = myia.basics.user_switch(#5, f:for:body, f:for:else)
  #7 = #6(phi_n, phi_acc)
  return #7
}

graph f:for:else(phi_n~2, phi_acc~2) {
  #8 = f:for_after(phi_acc~2)
  return #8
}

graph f:for_after(phi_acc~3) {
  return phi_acc~3
}

graph f:for:body(phi_n~3, phi_acc~4) {
  #9 = myia.basics.myia_next(it)
  i = _operator.getitem(#9, 0)
  #10 = _operator.getitem(#9, 1)
  #11 = myia.basics.resolve(:tests.test_parser, 'range')
  #12 = #11(phi_n~3)
  #13 = myia.basics.myia_iter(#12)
  #14 = f:for:body:for(#13, phi_n~3, phi_acc~4)
  return #14
}

graph f:for:body:for(it~2, phi_n~4, phi_acc~5) {
  #15 = myia.basics.myia_hasnext(it~2)
  #16 = myia.basics.user_switch(#15, f:for:body:for:body, f:for:body:for:else)
  #17 = #16(phi_n~4, phi_acc~5)
  return #17
}

graph f:for:body:for:else(phi_n~5, phi_acc~6) {
  #18 = f:for:body:for_after(phi_n~5, phi_acc~6)
  return #18
}

graph f:for:body:for_after(phi_n~6, phi_acc~7) {
  #19 = f:for(#10, phi_n~6, phi_acc~7)
  return #19
}

graph f:for:body:for:body(phi_n~7, phi_acc~8) {
  #20 = myia.basics.myia_next(it~2)
  j = _operator.getitem(#20, 0)
  #21 = _operator.getitem(#20, 1)
  acc = _operator.add(phi_acc~8, j)
  #22 = f:for:body:for(#21, phi_n~7, acc)
  return #22
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
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(x, y)
  return #3
}

graph f:if_false(phi_x, phi_y) {
  return phi_y
}

graph f:if_true(phi_x~2, phi_y~2) {
  return phi_x~2
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
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(y)
  return #3
}

graph f:if_false(phi_y) {
  #4 = f:if_after(phi_y)
  return #4
}

graph f:if_after(phi_y~2) {
  return phi_y~2
}

graph f:if_true(phi_y~3) {
  #5 = f:if_after(0)
  return #5
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
  #1 = f:while(b, x, y)
  return #1
}

graph f:while(phi_b, phi_x, phi_y) {
  #2 = myia.basics.user_switch(phi_b, f:while:body, f:while:else)
  #3 = #2(phi_x, phi_y)
  return #3
}

graph f:while:else(phi_x~2, phi_y~2) {
  #4 = f:while_after(phi_y~2)
  return #4
}

graph f:while_after(phi_y~3) {
  return phi_y~3
}

graph f:while:body(phi_x~3, phi_y~4) {
  return phi_x~3
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
  #1 = f:while(x)
  return #1
}

graph f:while(phi_x) {
  #2 = myia.basics.user_switch(phi_x, f:while:body, f:while:else)
  #3 = #2(phi_x)
  return #3
}

graph f:while:else(phi_x~2) {
  #4 = f:while_after(phi_x~2)
  return #4
}

graph f:while_after(phi_x~3) {
  return phi_x~3
}

graph f:while:body(phi_x~4) {
  x~2 = _operator.sub(phi_x~4, 1)
  #5 = f:while(x~2)
  return #5
}
"""
        )


class _ContextExample:
    def __enter__(self):
        print("Enter")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exit")


class _AsyncContextExample:
    async def __aenter__(self):
        print("Async enter")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Async exit")


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Import not supported"
)
def test_import():
    # ast.Import
    def f():  # pragma: no cover
        import operator

        return operator.add(1, 2)

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="ImportFrom not supported"
)
def test_import_from():
    # ast.ImportFrom
    def f():  # pragma: no cover
        from operator import add

        return add(1, 2)

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="ImportFrom not supported"
)
def test_import_as():
    # ast.alias
    def f():  # pragma: no cover
        from operator import add as addition

        return addition(1, 2)

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="AugAssign not supported"
)
def test_augmented_assignment():
    # ast.AugAssign
    # ast.Store as name ctx
    def f(x):  # pragma: no cover
        a = x
        a += 2
        return a

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="ClassDef not supported"
)
def test_class_def():
    # ast.ClassDef
    def f():  # pragma: no cover
        class A:
            pass

        return A()

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Delete not supported"
)
def test_del():
    # ast.Delete
    # ast.Del as name ctx
    def f(a, b):  # pragma: no cover
        del a
        return b

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Raise not supported"
)
def test_raise():
    # ast.Raise
    def f():  # pragma: no cover
        raise ValueError()

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Try not supported"
)
def test_try_block():
    # ast.Try
    # ast.ExceptHandler
    def f(x):  # pragma: no cover
        try:
            if x < 0:
                raise ValueError()
        except ValueError:
            pass
        finally:
            pass

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="With not supported"
)
def test_with_statement():
    # ast.With
    # ast.withitem
    def f():  # pragma: no cover
        with _ContextExample():
            return 0

    parse(f)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Parser expected FunctionDef, not AsyncFunctionDef",
)
def test_async_function():
    # ast.AsyncFunctionDef
    async def f():  # pragma: no cover
        pass

    parse(f)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Parser expected FunctionDef, not AsyncFunctionDef",
)
def test_async_await():
    # ast.Await
    async def f():  # pragma: no cover
        async def g():
            pass

        await g()

    parse(f)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Parser expected FunctionDef, not AsyncFunctionDef",
)
def test_async_for():
    # ast.AsyncFor
    async def f():  # pragma: no cover
        async def g():
            yield 0
            yield 1

        async for x in g():
            pass

    parse(f)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Parser expected FunctionDef, not AsyncFunctionDef",
)
def test_async_with():
    # ast.AsyncWith
    async def f():  # pragma: no cover
        async with _AsyncContextExample():
            pass

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="DictComp not supported"
)
def test_dict_comprehension():
    # ast.DictComp
    def f():  # pragma: no cover
        return {i: 2 * i for i in range(10)}

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="GeneratorExp not supported"
)
def test_generator_expression():
    # ast.GeneratorExp
    # ast.comprehension (used in all comprehensions)
    def f():  # pragma: no cover
        return list(2 * i for i in range(10))

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="ListComp not supported"
)
def test_list_comprehension():
    # ast.ListComp
    def f():  # pragma: no cover
        return [2 * i for i in range(10)]

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="SetComp not supported"
)
def test_set_comprehension():
    # ast.SetComp
    def f():  # pragma: no cover
        return {2 * i for i in range(10)}

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Set not supported"
)
def test_set():
    # ast.Set
    def f():  # pragma: no cover
        return {1, 2}

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="Yield not supported"
)
def test_yield():
    # ast.Yield
    def f():  # pragma: no cover
        def g():
            yield 0
            yield 1

        return list(g())

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="YieldFrom not supported"
)
def test_yield_from():
    # ast.YieldFrom
    def f():  # pragma: no cover
        def g():
            yield from range(10)

        return list(g())

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="NameExpr not supported"
)
def test_assignment_expression():
    # ast.NamedExpr
    def f(x):  # pragma: no cover
        if (a := 2) != x:
            return a

    parse(f)


@pytest.mark.xfail(
    strict=True, raises=MyiaSyntaxError, reason="JoinedStr not supported"
)
def test_formatted_string():
    # ast.JoinedStr
    # ast.FormattedValue
    def f(x):  # pragma: no cover
        return f"value is {x}"

    parse(f)


# NB: Are these nodes still used ?
# ast.AugLoad
# ast.AugStore
# ast.Param (Python 2 only?)


def test_branch_defined():
    def f(b):  # pragma: no cover
        def g():
            return a

        if b:
            a = 1
        else:
            a = 2
        return g()

    with enable_debug(), predictable_placeholders():
        assert (
            str_graph(parse(f))
            == """graph f(b) {
  a = myia.basics.make_handle(??1)
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(g)
  return #3
}

graph f:if_false(phi_g) {
  #4 = myia.basics.global_universe_setitem(a, 2)
  #5 = f:if_after(phi_g)
  return #5
}

graph f:if_after(phi_g~2) {
  #6 = phi_g~2()
  return #6
}

graph f:if_true(phi_g~3) {
  #7 = myia.basics.global_universe_setitem(a, 1)
  #8 = f:if_after(phi_g~3)
  return #8
}

graph g() {
  #9 = myia.basics.global_universe_getitem(a)
  return #9
}
"""
        )


def test_branch_defined2():
    def f(b, x, y):  # pragma: no cover
        def g():
            return a

        if b:
            a = x - y
        else:
            a = x + y
        return g()

    with enable_debug(), predictable_placeholders():
        assert (
            str_graph(parse(f))
            == """graph f(b, x, y) {
  a = myia.basics.make_handle(??1)
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(x, y, g)
  return #3
}

graph f:if_false(phi_x, phi_y, phi_g) {
  #4 = _operator.add(phi_x, phi_y)
  #5 = myia.basics.global_universe_setitem(a, #4)
  #6 = f:if_after(phi_g)
  return #6
}

graph f:if_after(phi_g~2) {
  #7 = phi_g~2()
  return #7
}

graph f:if_true(phi_x~2, phi_y~2, phi_g~3) {
  #8 = _operator.sub(phi_x~2, phi_y~2)
  #9 = myia.basics.global_universe_setitem(a, #8)
  #10 = f:if_after(phi_g~3)
  return #10
}

graph g() {
  #11 = myia.basics.global_universe_getitem(a)
  return #11
}
"""
        )


def test_branch_defined3():
    def f(b, c):  # pragma: no cover
        def g():
            return a

        if b:
            if c:
                a = 2
            else:
                a = 3
        else:
            if c:
                a = 1
            else:
                a = 0
        return g()

    with enable_debug(), predictable_placeholders():
        assert (
            str_graph(parse(f))
            == """graph f(b, c) {
  a = myia.basics.make_handle(??1)
  #1 = _operator.truth(b)
  #2 = myia.basics.user_switch(#1, f:if_true, f:if_false)
  #3 = #2(c, g)
  return #3
}

graph f:if_false(phi_c, phi_g) {
  #4 = _operator.truth(phi_c)
  #5 = myia.basics.user_switch(#4, f:if_false:if_true, f:if_false:if_false)
  #6 = #5(phi_g)
  return #6
}

graph f:if_false:if_false(phi_g~2) {
  #7 = myia.basics.global_universe_setitem(a, 0)
  #8 = f:if_false:if_after(phi_g~2)
  return #8
}

graph f:if_false:if_after(phi_g~3) {
  #9 = f:if_after(phi_g~3)
  return #9
}

graph f:if_after(phi_g~4) {
  #10 = phi_g~4()
  return #10
}

graph f:if_false:if_true(phi_g~5) {
  #11 = myia.basics.global_universe_setitem(a, 1)
  #12 = f:if_false:if_after(phi_g~5)
  return #12
}

graph f:if_true(phi_c~2, phi_g~6) {
  #13 = _operator.truth(phi_c~2)
  #14 = myia.basics.user_switch(#13, f:if_true:if_true, f:if_true:if_false)
  #15 = #14(phi_g~6)
  return #15
}

graph f:if_true:if_false(phi_g~7) {
  #16 = myia.basics.global_universe_setitem(a, 3)
  #17 = f:if_true:if_after(phi_g~7)
  return #17
}

graph f:if_true:if_after(phi_g~8) {
  #18 = f:if_after(phi_g~8)
  return #18
}

graph f:if_true:if_true(phi_g~9) {
  #19 = myia.basics.global_universe_setitem(a, 2)
  #20 = f:if_true:if_after(phi_g~9)
  return #20
}

graph g() {
  #21 = myia.basics.global_universe_getitem(a)
  return #21
}
"""
        )


@pytest.mark.xfail()
def test_cursed_function():
    def f(b):  # pragma: no cover
        a = 0  # noqa: F841

        def g(b):
            def h():
                return a

            if b:
                a = 1
            return h()

        return g(b)

    with enable_debug():
        assert (
            str_graph(parse(f))
            == """graph f(b) {
  #1 = g(b)
  return #1
}

graph g(b~2) {
  a = myia.basics.make_handle(??1)
  #2 = _operator.truth(b~2)
  #3 = myia.basics.user_switch(#2, g:if_true, g:if_false)
  #4 = #3(h)
  return #4
}

graph g:if_false(phi_h) {
  #5 = g:if_after(phi_h)
  return #5
}

graph g:if_after(phi_h~2) {
  #6 = phi_h~2()
  return #6
}

graph g:if_true(phi_h~3) {
  #7 = myia.basics.global_universe_setitem(a, 1)
  #8 = g:if_after(phi_h~3)
  return #8
}

graph h() {
  #9 = myia.basics.global_universe_getitem(a)
  return #9
}
"""
        )
