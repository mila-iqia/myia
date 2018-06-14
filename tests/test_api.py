import pytest

from myia.api import parse, compile
from myia.cconv import closure_convert
from myia.ir import clone
from myia.prim.py_implementations import getitem


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
