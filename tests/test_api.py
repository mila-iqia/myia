import numpy as np
import pytest

from myia.api import myia, convert_arg, convert_result, \
    scalar_parse as parse, scalar_debug_compile as compile
from myia.cconv import closure_convert
from myia.dtype import List, Array, Tuple, Bool
from myia.infer import InferenceError
from myia.ir import clone
from myia.prim.py_implementations import getitem

from .common import Point, Point_t, Point3D, i64, f64, i16


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
        f((10, 20), (30, 40))


def test_myia_specialize_values():
    @myia(specialize_values=['c'])
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


def test_convert_arg():

    # Leaves

    assert convert_arg(True, Bool) == [True]
    assert convert_arg(False, Bool) == [False]
    assert convert_arg(10, i64) == [10]
    assert convert_arg(1.5, f64) == [1.5]

    # Class -> Tuple conversion

    pt = Point(1, 2)
    pt3 = Point3D(1, 2, 3)
    assert list(convert_arg(pt, Point_t)) == [1, 2]
    with pytest.raises(TypeError):
        convert_arg((1, 2), Point_t)

    assert list(convert_arg((pt, pt),
                Tuple[Point_t, Point_t])) == [1, 2, 1, 2]

    assert convert_arg([pt, pt, pt],
                       List[Point_t]) == [[1, 2, 1, 2, 1, 2]]

    # Arrays

    fmat = np.ones((5, 8))
    imat = np.ones((5, 8), dtype='int16')

    assert convert_arg(fmat, Array[f64])[0] is fmat
    assert convert_arg(imat, Array[i16])[0] is imat
    with pytest.raises(TypeError):
        convert_arg(imat, Array[i64])

    # Misc errors

    with pytest.raises(TypeError):
        convert_arg(10, f64)
    with pytest.raises(TypeError):
        convert_arg(1.5, i64)
    with pytest.raises(TypeError):
        convert_arg(10, Tuple[i64, i64])
    with pytest.raises(TypeError):
        convert_arg((1,), Tuple[i64, i64])
    with pytest.raises(TypeError):
        convert_arg((1, 2, 3), Tuple[i64, i64])
    with pytest.raises(TypeError):
        convert_arg((1, 2, 3), List[i64])
    with pytest.raises(TypeError):
        convert_arg(pt3, Point_t)
    with pytest.raises(TypeError):
        convert_arg(10, Array[i64])
    with pytest.raises(TypeError):
        convert_arg(10, Array[i64])
    with pytest.raises(TypeError):
        convert_arg(1, Bool)


def test_convert_result():

    # Leaves

    assert convert_result(True, Bool, Bool) is True
    assert convert_result(False, Bool, Bool) is False
    assert convert_result(10, i64, i64) == 10
    assert convert_result(1.5, f64, f64) == 1.5

    # Tuple -> Class conversion

    pt = Point(1, 2)
    assert convert_result(pt, Point_t, Point_t) == pt
    assert convert_result((1, 2), Point_t, Tuple[i64, i64]) == pt

    assert convert_result(((1, 2), (1, 2)),
                          Tuple[Point_t, Point_t],
                          Tuple[Tuple[i64, i64], Tuple[i64, i64]]) == \
        (pt, pt)

    assert convert_result([(1, 2), (1, 2), (1, 2)],
                          List[Point_t],
                          List[Tuple[i64, i64]]) == \
        [pt, pt, pt]


def test_function_arg():
    """Give a Python function as an argument."""
    def square(x):  # pragma: no cover
        return x * x

    @compile
    def f(fn, x, y):
        return fn(x + y)

    assert f(square, 10, 5) == 225


def test_function_in_tuple():
    """Give a tuple of functions as an argument."""
    def square(x):  # pragma: no cover
        return x * x

    def double(x):  # pragma: no cover
        return x + x

    @compile
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
        return getitem

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
