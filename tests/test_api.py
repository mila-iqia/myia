import numpy as np
import pytest

from myia.abstract.data import AbstractRandomState
from myia.api import myia, to_device
from myia.compile import closure_convert
from myia.ir import clone
from myia.operations import tuple_getitem
from myia.pipeline import (
    scalar_debug_compile as compile,
    scalar_parse as parse,
)
from myia.simplify_types import from_canonical, to_canonical
from myia.utils import DoTrace, HandleInstance, InferenceError, TaggedValue
from myia.utils.misc import RandomStateWrapper
from myia.xtype import Bool

from .common import (
    D,
    Point,
    Point3D,
    Thing,
    af64_of,
    ai32_of,
    ai64_of,
    f64,
    i64,
    to_abstract_test,
)


def test_myia():
    @myia
    def f(x, y):
        return x + y

    assert f(10, 20) == 30

    fi = f.compile((10, 20))
    ff = f.compile((10.0, 20.0))
    assert fi is not ff
    assert fi is f.compile((100, 200))

    with pytest.raises(InferenceError):
        f(10)
    with pytest.raises(InferenceError):
        f(10, 20, 30)
    with pytest.raises(InferenceError):
        f(Thing(10), Thing(20))


def test_myia_specialize_values():
    @myia(specialize_values=["c"])
    def f(c, x, y):
        if c:
            return x + y
        else:
            return x * y

    assert f(True, 10, 20) == 30
    assert f(False, 10, 20) == 200

    ft = f.compile((True, 10, 20))
    ff = f.compile((False, 10, 20))
    assert ft is not ff


def test_myia_struct_arg():
    @myia
    def f(pt):
        return pt.x

    x = f(Point(5, 6))
    assert x == 5


def test_myia_return_struct():
    @myia
    def f(x, y):
        return Point(x, y)

    pt = f(5, 6)
    assert pt == Point(5, 6)


def test_myia_dict_field():
    @myia
    def f(d):
        return d["x"]

    v = f({"x": 2})
    assert v == 2


def test_random_state_to_canonical():
    rsw = RandomStateWrapper(None)
    ars = AbstractRandomState()
    assert to_canonical(rsw, ars) is rsw
    with pytest.raises(TypeError):
        to_canonical(None, ars)


def test_to_canonical():
    def _convert(data, typ):
        return to_canonical(data, to_abstract_test(typ))

    # Leaves

    assert _convert(True, Bool) is True
    assert _convert(False, Bool) is False
    assert _convert(10, i64) == 10
    assert _convert(1.5, f64) == 1.5
    assert _convert([], []) == ()
    with pytest.raises(TypeError):
        _convert([], [f64])
    with pytest.raises(TypeError):
        _convert([1, 2], [])

    # Class -> Tuple conversion

    pt = Point(1, 2)
    pt3 = Point3D(1, 2, 3)
    assert list(_convert(pt, Point(i64, i64))) == [1, 2]
    with pytest.raises(TypeError):
        _convert((1, 2), Point(i64, i64))

    assert list(_convert((pt, pt), (Point(i64, i64), Point(i64, i64)))) == [
        (1, 2),
        (1, 2),
    ]

    li = _convert([1], [i64])
    assert (
        isinstance(li, tuple)
        and li[0] == 1
        and isinstance(li[1], TaggedValue)
        and li[1].value == ()
    )

    # Arrays

    fmat = np.ones((5, 8))
    imat = np.ones((5, 8), dtype="int32")

    assert _convert(fmat, af64_of(5, 8)) is fmat
    assert _convert(imat, ai32_of(5, 8)) is imat
    with pytest.raises(TypeError):
        _convert(imat, ai64_of(5, 8))
    with pytest.raises(TypeError):
        _convert(imat, ai32_of(4, 8))

    # Misc errors

    with pytest.raises(TypeError):
        _convert(10, f64)
    with pytest.raises(TypeError):
        _convert(1.5, i64)
    with pytest.raises(TypeError):
        _convert(10, (i64, i64))
    with pytest.raises(TypeError):
        _convert((1,), (i64, i64))
    with pytest.raises(TypeError):
        _convert((1, 2, 3), (i64, i64))
    with pytest.raises(TypeError):
        _convert((1, 2, 3), [i64])
    with pytest.raises(TypeError):
        _convert(pt3, Point(i64, i64))
    with pytest.raises(TypeError):
        _convert(10, ai64_of())
    with pytest.raises(TypeError):
        _convert(10, ai64_of())
    with pytest.raises(TypeError):
        _convert(1, Bool)
    with pytest.raises(TypeError):
        _convert(1, D(x=i64))
    with pytest.raises(TypeError):
        _convert({"x": 2.0}, D(x=i64))
    with pytest.raises(TypeError):
        _convert({"x": 2.0, "y": 1}, D(x=i64))
    with pytest.raises(TypeError):
        _convert({"y": 2.0}, D(x=i64))
    with pytest.raises(TypeError):
        _convert("x", 1.0)
    with pytest.raises(TypeError):
        _convert(1.0, to_abstract_test("x"))
    with pytest.raises(TypeError):
        _convert(1.0, to_abstract_test(HandleInstance(1.0)))

    v = to_device(22, None)
    with pytest.raises(TypeError):
        _convert(v, f64)


def test_from_canonical():
    def _convert(data, typ):
        return from_canonical(data, to_abstract_test(typ))

    # Leaves

    assert _convert(True, Bool) is True
    assert _convert(False, Bool) is False
    assert _convert(10, i64) == 10
    assert _convert(1.5, f64) == 1.5

    # Tuple -> Class conversion

    pt = Point(1, 2)
    assert _convert((1, 2), Point(i64, i64)) == pt
    assert _convert(((1, 2), (1, 2)), (Point(i64, i64), Point(i64, i64))) == (
        pt,
        pt,
    )


@pytest.mark.xfail(reason="Cannot pass a function as an argument.")
def test_function_arg():
    """Give a Python function as an argument."""

    def square(x):
        return x * x

    @myia
    def f(fn, x, y):
        return fn(x + y)

    assert f(square, 10, 5) == 225


@pytest.mark.xfail(reason="Cannot pass a function as an argument.")
def test_function_in_tuple():
    """Give a tuple of functions as an argument."""

    def square(x):
        return x * x

    def double(x):
        return x + x

    @myia
    def f(fns, x, y):
        f0, f1 = fns
        return f1(f0(x + y))

    assert f((square, double), 10, 5) == 450
    assert f((double, square), 10, 5) == 900


def test_return_closure():
    """Return a closure."""

    @compile
    def f(x, y):
        def g():
            return x + y

        return g

    assert f(4, 5)() == 9


def test_return_closure_partial():
    """Return a closure (after closure conversion)."""

    @parse
    def f(x, y):
        def g():
            return x + y

        return g

    f = clone(closure_convert(f))
    f = compile(f)

    g = f(4, 5)
    assert g() == 9


def test_return_closure_tuple():
    """Return a tuple of closures."""

    @compile
    def f(x, y):
        def g():
            return x + y

        def h():
            return x * y

        return (g, h)

    g, h = f(4, 5)
    assert g() == 9
    assert h() == 20


def test_refeed():
    """Return a closure, then use the closure as an argument."""

    @compile
    def f(fn, x, y):
        def g():
            return x + y

        if x == 0:
            return g
        else:
            return fn()

    g = f(None, 0, 6)
    assert g() == 6
    assert f(g, 10, 20) == 6


def test_return_primitive():
    """Return a primitive."""

    @compile
    def f():
        return tuple_getitem

    g = f()
    assert g((1, 2, 3), 0) == 1


def test_return_graph():
    @compile
    def f():
        def g():
            return 42

        return g

    g = f()
    assert g() == 42


def test_bad_call1():
    @compile
    def f():
        return 42

    with pytest.raises(RuntimeError):
        f(1)


def test_bad_call2():
    @compile
    def f():
        def g():
            return 42

        return g(0)

    with pytest.raises(RuntimeError):
        f()


def test_tail_call():
    @compile
    def f():
        x = 1000
        while x > 0:
            x = x - 1
        return x


def test_raise():
    @compile
    def f(x):
        if x == 0:
            return 1
        elif x >= 10:
            raise Exception("too big")
        else:
            return x * f(x - 1)

    assert f(5) == 120

    try:
        f(10)
    except Exception as exc:
        assert type(exc) is Exception
        assert exc.args == ("too big",)
    else:
        raise Exception("Expected an exception")


def test_return_backend():
    @myia(return_backend=True)
    def f(a):
        return a

    f(33)


_flag1 = [False]
_flag2 = [False]


def _setflag1():
    def _set(**_):
        _flag1[0] = True

    return DoTrace({"/compile/enter": _set})


def _setflag2():
    def _set(**_):
        _flag2[0] = True

    return DoTrace({"/compile/exit": _set})


def test_env_tracer(monkeypatch):
    _flag1[0] = False
    _flag2[0] = False

    monkeypatch.setenv(
        "MYIATRACER", "tests.test_api._setflag1;tests.test_api._setflag2"
    )

    @myia
    def f(x, y):
        return x * y

    assert f(10, 11) == 110

    assert _flag1[0] is True
    assert _flag2[0] is True
