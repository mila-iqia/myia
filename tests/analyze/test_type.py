import pytest

from myia.ir.anf import Graph
from myia.api import parse
from myia.dtype import Function, Bool, Int, Float, Tuple
from myia.prim.py_implementations import cons_tuple

from myia.analyze.graph import GraphAnalyzer


def test_infer_args():
    def f(x, y):
        return x + y

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    equiv = dict(ga._equiv)
    assert ga.infer_args(gf, (2, 3)) == Int(64)
    assert equiv == ga._equiv

    assert ga.infer_args(gf, (1.0, 1.0)) == Float(64)
    assert equiv == ga._equiv

    with pytest.raises(TypeError):
        ga.infer_args(gf, (2, 1.0))
    assert equiv == ga._equiv

    with pytest.raises(TypeError):
        ga.infer_args(gf, (1.0, 3))
    assert equiv == ga._equiv

    with pytest.raises(TypeError):
        ga.infer_args(gf, (1.0,))
    assert equiv == ga._equiv

    with pytest.raises(TypeError):
        ga.infer_args(gf, (True, False))
    assert equiv == ga._equiv


def test_infer_type():
    ga = GraphAnalyzer()

    with pytest.raises(ValueError):
        ga.infer_type(Graph(), None)


def test_base():
    def f(x):
        return x

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    gf_t = ga.signatures[gf]
    assert gf_t.arguments[0] is gf_t.retval


def test_base2():
    def f(x):
        return x + x

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    gf_t = ga.signatures[gf]
    assert gf_t.arguments[0] is gf_t.retval
    assert ga.infer_args(gf, (1,)) is Int(64)
    assert ga.infer_args(gf, (1.0,)) is Float(64)
    with pytest.raises(TypeError):
        ga.infer_type(gf, (Bool(),))


def test_infer():
    def g(x, y):
        if x == 0:
            return y
        else:
            return x

    gg = parse(g)
    ga = GraphAnalyzer()
    ga.analyze(gg)
    assert ga.signatures[gg] == Function((Int(64), Int(64)), Int(64))


def test_bad_call():
    def f(x, y):
        return y

    def g(x):
        return f(x)

    gg = parse(g)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gg)


def test_bad_call2():
    def g(x):
        return x

    def f():
        return g(1)()

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_bad_call3():
    def f(x):
        def g(fn):
            return fn(1.0)

        def h(fn):
            return fn(1)
        return h(x) + g(x)

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_if():
    def f():
        if False:
            return 0
        else:
            return 1.0

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    gf_t = ga.signatures[gf]
    assert gf_t is Function((), Float(64))


def test_if2():
    def f(x):
        if x:
            return -1
        else:
            return 1.0

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_recur():
    def f(x, y):
        if x != 0:
            return f(0, 33)
        else:
            return y

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    gf_t = ga.signatures[gf]
    assert gf_t is Function((Int(64), Int(64)), Int(64))


def test_fact():
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    gfact = parse(fact)
    ga = GraphAnalyzer()
    ga.analyze(gfact)

    assert ga.signatures[gfact] is Function((Int(64),), Int(64))


def test_pow10():
    def pow10(x):
        v = x
        j = 0
        while j < 3:
            i = 0
            while i < 3:
                v = v * x
                i = i + 1
            j = j + 1
        return v

    gpow10 = parse(pow10)
    ga = GraphAnalyzer()
    ga.analyze(gpow10)
    pow10_t = ga.signatures[gpow10]
    assert pow10_t.arguments[0] is pow10_t.retval
    assert len(pow10_t.arguments) == 1
    assert pow10_t.retval.matches(Int(64))
    assert pow10_t.retval.matches(Float(64))
    assert not pow10_t.retval.matches(Bool())


def test_func_arg():
    def f(x, y):
        def g(func, x, y):
            return func(x, y)

        def h(x, y):
            return x + y
        return g(h, x, y)

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    g_t = ga.signatures[gf]
    assert g_t.arguments[0] is g_t.arguments[1]
    assert g_t.arguments[0] is g_t.retval
    assert ga.infer_args(gf, (1, 2)) is Int(64)

    with pytest.raises(TypeError):
        assert ga.infer_args(gf, (True, False))


def test_func_arg2():
    def f(x, y):
        def g(func, x, y):
            return func(x, y)
        return g(1, x, y)

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_func_arg3():
    def f(x):
        def g(func, x):
            z = func + x
            return func(z)

        def h(x):
            return x

        return g(h, x)

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_func_arg4():
    def f(x):
        def h(x):
            return x

        def g(fn, x):
            return fn(h, x)

        def t(fn, x):
            return fn(x)

        return g(t, x)

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    g_t = ga.signatures[gf]
    assert g_t.arguments[0] is g_t.retval


def test_func_arg5():
    def f(x):
        def g(fn):
            return fn(-1)

        def h(fn):
            return fn(1)
        return h(x) + g(x)

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    f_t = ga.signatures[gf]
    # The argument of fn is Int(64)
    assert f_t.arguments[0].arguments[0] is Int(64)


def test_closure():
    def f():
        def g(x):
            def h():
                return x
            return h
        return g(2)()

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    f_t = ga.signatures[gf]
    assert f_t is Function((), Int(64))


def test_cons_tuple():
    def g(e, t):
        return cons_tuple(e, t)

    def f(a):
        return g(a, (1, 2.0, False))

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    g_t = ga.signatures[gf]

    assert isinstance(g_t.retval, Tuple)
    assert g_t.arguments[0] is g_t.retval.elements[0]
    assert g_t.retval.elements[1] is Int(64)
    assert g_t.retval.elements[2] is Float(64)
    assert g_t.retval.elements[3] is Bool()


def test_getitem():
    def f(i):
        xt = (1, 2.0, False)
        return xt[0]

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    f_t = ga.signatures[gf]
    assert f_t.retval is Int(64)
