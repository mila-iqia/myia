import pytest

from myia.anf_ir import Graph
from myia.api import parse
from myia.dtype import Function, Bool, Int, Float, Tuple, Struct, List
from myia.prim import ops as P
from myia.prim.py_implementations import cons_tuple
from myia.unify import var

from myia.analyze.graph import GraphAnalyzer
from myia.analyze.type import (typeof, TypePlugin,
                               TypeInferenceError)


def test_typeof():
    assert Int(64) is typeof(1)
    assert Bool() is typeof(True)
    assert Tuple(Bool(), Int(64)) is typeof((False, 22))
    assert isinstance(typeof(P.add), Function)
    with pytest.raises(TypeError):
        typeof(object())


def test_bad_node():
    t = TypePlugin()
    with pytest.raises(AssertionError):
        t.on_node(None)


def test_visit():
    ga = GraphAnalyzer([TypePlugin()])
    U = ga.U
    v1 = var()
    v2 = var()

    l1 = List(v1)
    l2 = U.clone(l1)
    assert l1 is not l2
    assert U.unify(l1, l2)[v1] is l2.element_type

    s1 = Struct(a=v1, b=v2)
    s2 = U.clone(s1)
    assert s1 is not s2
    assert set(s2.elements.keys()) == {'a', 'b'}

    t1 = Tuple(v1, v1, v2)
    t2 = U.clone(t1)
    assert t1 is not t2
    assert t2.elements[0] is t2.elements[1]
    assert t2.elements[0] is not t2.elements[2]
    assert len(t2.elements) == 3

    c1 = Function((v1, v2), v2)
    c2 = U.clone(c1)
    assert c1 is not c2
    assert c2.arguments[1] is c2.retval
    assert c2.arguments[0] is not c2.arguments[1]
    assert len(c2.arguments) == 2

    b = U.clone(Bool())
    assert b is Bool()


def test_infer_args():
    def f(x, y):
        return x + y

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    equiv = dict(ga.equiv)
    assert ga.infer_args(gf, 2, 3) == Int(64)
    assert equiv == ga.equiv

    assert ga.infer_args(gf, 1.0, 1.0) == Float(64)
    assert equiv == ga.equiv

    with pytest.raises(TypeInferenceError):
        ga.infer_args(gf, 2, 1.0)
    assert equiv == ga.equiv

    with pytest.raises(TypeInferenceError):
        ga.infer_args(gf, 1.0, 3)
    assert equiv == ga.equiv

    with pytest.raises(TypeInferenceError):
        ga.infer_args(gf, True, False)
    assert equiv == ga.equiv


def test_infer_type():
    ga = GraphAnalyzer([TypePlugin()])

    with pytest.raises(ValueError):
        ga.infer_type(Graph())


def test_base():
    def f(x):
        return x

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    gf_t = ga.graphs[gf]['type']
    assert gf_t.arguments[0] is gf_t.retval


def test_base2():
    def f(x):
        return x + x

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    gf_t = ga.graphs[gf]['type']
    assert gf_t.arguments[0] is gf_t.retval
    assert ga.infer_args(gf, 1) is Int(64)
    assert ga.infer_args(gf, 1.0) is Float(64)
    with pytest.raises(TypeInferenceError):
        ga.infer_type(gf, Bool())


def test_infer():
    def g(x, y):
        if x == 0:
            return y
        else:
            return x

    gg = parse(g)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gg)
    assert ga.graphs[gg]['type'] == Function((Int(64), Int(64)), Int(64))


def test_bad_call():
    def f(x, y):
        return y

    def g(x):
        return f(x)

    gg = parse(g)
    ga = GraphAnalyzer([TypePlugin()])
    with pytest.raises(TypeInferenceError):
        ga.analyze(gg)


def test_recur():
    def f(x, y):
        if x != 0:
            return f(0, 33)
        else:
            return y

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    gf_t = ga.graphs[gf]['type']
    assert gf_t is Function((Int(64), Int(64)), Int(64))


def test_fact():
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    gfact = parse(fact)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gfact)

    assert ga.graphs[gfact]['type'] is Function((Int(64),), Int(64))


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
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gpow10)
    pow10_t = ga.graphs[gpow10]['type']
    assert pow10_t.arguments[0] is pow10_t.retval
    assert len(pow10_t.arguments) == 1
    assert pow10_t.retval.matches(Int(64))
    assert pow10_t.retval.matches(Float(64))
    assert not pow10_t.retval.matches(Bool())


def test_multi_fn():
    def g(x, y):
        return x + y

    def f(x, y, z):
        if x == g(y, -1):
            return g(z, 2.0)
        else:
            return g(z, -2.0)

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    assert ga.graphs[gf]['type'] == Function((Int(64), Int(64), Float(64)),
                                             Float(64))


def test_func_arg():
    def f(x, y):
        def g(func, x, y):
            return func(x, y)

        def h(x, y):
            return x + y
        return g(h, x, y)

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    g_t = ga.graphs[gf]['type']
    assert g_t.arguments[0] is g_t.arguments[1]
    assert g_t.arguments[0] is g_t.retval
    assert ga.infer_args(gf, 1, 2) is Int(64)

    with pytest.raises(TypeInferenceError):
        assert ga.infer_args(gf, True, False)


def test_func_arg2():
    def f(x, y):
        def g(func, x, y):
            return func(x, y)
        return g(1, x, y)

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    with pytest.raises(TypeInferenceError):
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
    ga = GraphAnalyzer([TypePlugin()])
    with pytest.raises(TypeInferenceError):
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
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    g_t = ga.graphs[gf]['type']
    assert g_t.arguments[0] is g_t.retval


def test_cons_tuple():
    def g(e, t):
        return cons_tuple(e, t)

    def f(a):
        return g(a, (1, 2.0, False))

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    g_t = ga.graphs[gf]['type']

    assert isinstance(g_t.retval, Tuple)
    assert g_t.arguments[0] is g_t.retval.elements[0]
    assert g_t.retval.elements[1] is Int(64)
    assert g_t.retval.elements[2] is Float(64)
    assert g_t.retval.elements[3] is Bool()


def Xtest_getitem():
    def f(i):
        x = (1, 2.0, False)
        return x[0]

    gf = parse(f)
    ga = GraphAnalyzer([TypePlugin()])
    ga.analyze(gf)

    f_t = ga.graphs[gf]['type']
    assert f_t.retval is Int(64)
