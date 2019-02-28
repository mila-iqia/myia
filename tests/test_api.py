import numpy as np
import pytest

from myia.api import myia
from myia.cconv import closure_convert
from myia.dtype import Bool
from myia.abstract import InferenceError
from myia.ir import clone
from myia.pipeline import \
    scalar_parse as parse, scalar_debug_compile as compile
from myia.pipeline.steps import convert_arg, convert_result
from myia.prim.py_implementations import tuple_getitem

from .common import Point, Point3D, i64, f64, to_abstract_test, ai64_of, \
    ai32_of, af64_of


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

    def _convert(data, typ):
        return convert_arg(data, to_abstract_test(typ))

    # Leaves

    assert _convert(True, Bool) == [True]
    assert _convert(False, Bool) == [False]
    assert _convert(10, i64) == [10]
    assert _convert(1.5, f64) == [1.5]

    # Class -> Tuple conversion

    pt = Point(1, 2)
    pt3 = Point3D(1, 2, 3)
    assert list(_convert(pt, Point(i64, i64))) == [1, 2]
    with pytest.raises(TypeError):
        _convert((1, 2), Point(i64, i64))

    assert list(_convert((pt, pt),
                (Point(i64, i64), Point(i64, i64)))) == [1, 2, 1, 2]

    assert _convert([pt, pt, pt],
                    [Point(i64, i64)]) == [[1, 2, 1, 2, 1, 2]]

    # Arrays

    fmat = np.ones((5, 8))
    imat = np.ones((5, 8), dtype='int32')

    assert _convert(fmat, af64_of(5, 8))[0] is fmat
    assert _convert(imat, ai32_of(5, 8))[0] is imat
    with pytest.raises(TypeError):
        _convert(imat, ai64_of(5, 8))

    # Misc errors

    with pytest.raises(TypeError):
        _convert(10, f64)
    with pytest.raises(TypeError):
        _convert("blah", to_abstract_test("blah"))
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


def test_convert_result():

    def _convert(data, typ1, typ2):
        return convert_result(data,
                              to_abstract_test(typ1),
                              to_abstract_test(typ2))

    # Leaves

    assert _convert(True, Bool, Bool) is True
    assert _convert(False, Bool, Bool) is False
    assert _convert(10, i64, i64) == 10
    assert _convert(1.5, f64, f64) == 1.5

    # Tuple -> Class conversion

    pt = Point(1, 2)
    assert _convert(pt, Point(i64, i64), Point(i64, i64)) == pt
    assert _convert((1, 2), Point(i64, i64), (i64, i64)) == pt

    assert _convert(((1, 2), (1, 2)),
                    (Point(i64, i64), Point(i64, i64)),
                    ((i64, i64), (i64, i64))) == \
        (pt, pt)

    assert _convert([(1, 2), (1, 2), (1, 2)],
                    [Point(i64, i64)],
                    [(i64, i64)]) == \
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
