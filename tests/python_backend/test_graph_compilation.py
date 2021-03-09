from dataclasses import dataclass

from myia.compile.backends.python.python import compile_graph
from myia.ir import manage
from myia.ir.utils import print_graph
from myia.operations import (
    grad,
    random_initialize,
    random_uint32,
    value_and_grad,
)
from myia.parser import parse


def _assert_match(expected, given, rel=1e-03):
    """Assert two values match.

    Use to check gradient output (computed using finite difference).
    Inspired to gradient output checking in myia.debug.finite_diff.GradTester.
    """
    threshold = max(abs(rel * expected), abs(rel * given))
    match = bool(abs(expected - given) <= threshold)
    assert match, (expected, given, rel)


def parse_and_compile(fn):
    g = parse(fn)
    manage(g)
    print(print_graph(g))
    cf = compile_graph(g, debug=True)
    return cf


def test_grad_first_order():
    def square(x):
        return x * x

    @parse_and_compile
    def f(x):
        return grad(square)(x)

    _assert_match(20, f(10))


def test_grad_two_args():
    def func(x, y):
        return 2 * x * y + 2 * x * x - 3 * y * y

    @parse_and_compile
    def f(x, y):
        return grad(func, "y")(x, y)

    @parse_and_compile
    def g(x, y):
        return grad(func, "x")(x, y)

    @parse_and_compile
    def h(x, y):
        return grad(func)(x, y)

    _assert_match(-10, f(1, 2))
    _assert_match(8, g(1, 2))

    dx, dy = h(2, 3)
    _assert_match(2 * 3 + 4 * 2, dx)
    _assert_match(2 * 2 - 6 * 3, dy)


def test_value_and_grad_first_order():
    def square(x):
        return x * x

    @parse_and_compile
    def f(x):
        return value_and_grad(square)(x)

    v, g = f(10)
    assert v == 100, v
    _assert_match(20, g)


def test_value_grad_two_args():
    def func(x, y):
        return 2 * x * y + 2 * x * x - 3 * y * y

    @parse_and_compile
    def f(x, y):
        return value_and_grad(func, "y")(x, y)

    @parse_and_compile
    def g(x, y):
        return value_and_grad(func, "x")(x, y)

    @parse_and_compile
    def h(x, y):
        return value_and_grad(func)(x, y)

    func_1_2 = -6
    assert func_1_2 == func(1, 2)

    value_f, grad_f = f(1, 2)
    assert value_f == func_1_2
    _assert_match(-10, grad_f)

    value_g, grad_g = g(1, 2)
    assert value_g == func_1_2
    _assert_match(8, grad_g)

    v, (dx, dy) = h(2, 3)
    assert v == 12 + 8 - 27
    _assert_match(2 * 3 + 4 * 2, dx)
    _assert_match(2 * 2 - 6 * 3, dy)


def test_rng():
    @parse_and_compile
    def f():
        rstate = random_initialize(12345678)
        r0, v0 = random_uint32(rstate, (2, 2))
        return v0

    print(f())


def test_while():
    @parse_and_compile
    def f(n):
        ret = 0
        i = 0
        while i < n + 1:
            ret = ret + i
            i = i + 1
        return ret

    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 3
    assert f(3) == 6


def test_iter_object():
    @dataclass
    class HalfIterator:
        value: int

        def __myia_iter__(self):
            return self

        def __myia_hasnext__(self):
            return self.value > 0

        def __myia_next__(self):
            return self.value, HalfIterator(self.value // 2)

    @parse_and_compile
    def f(v):
        ret = 0
        for x in HalfIterator(v):
            ret = ret + x
        return ret

    assert f(10) == 18, f(10)


def test_for_range():
    @parse_and_compile
    def f(n):
        ret = 0
        for i in range(n + 1):
            ret = ret + i
        return ret

    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 3
    assert f(3) == 6


def test_recursion():
    @parse_and_compile
    def factorial(n):
        return 1 if n < 2 else n * factorial(n - 1)

    for x, y in (
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
    ):
        assert factorial(x) == y, factorial(x)


def test_function_call():
    def _pgcd(a, b):
        return a if not b else _pgcd(b, a % b)

    @parse_and_compile
    def pgcd(a, b):
        return _pgcd(b, a) if a < b else _pgcd(a, b)

    assert pgcd(12, 32) == 4, pgcd(12, 32)
    assert pgcd(625, 250) == 125, pgcd(625, 250)


def test_inner_closure():
    @parse_and_compile
    def pgcd(a, b):
        def _pgcd(a, b):
            return a if not b else _pgcd(b, a % b)

        return _pgcd(b, a) if a < b else _pgcd(a, b)

    assert pgcd(12, 32) == 4, pgcd(12, 32)
    assert pgcd(625, 250) == 125, pgcd(625, 250)


def test_if():
    @parse_and_compile
    def f(x):
        if x < 0:
            return x * x
        else:
            return x * x * x

    assert f(-2) == 4, f(-2)
    assert f(0) == 0, f(0)
    assert f(2) == 8, f(2)


def test_simple():
    @parse_and_compile
    def f(x):
        return 2 * x + 1

    assert f(0) == 1, f(0)
    assert f(-2) == -3, f(0)


def test_two_external_functions_with_same_name():
    def f1(x):
        return 2 * x

    def f2(x):
        return 3 * x

    f1.__name__ = "f"
    f2.__name__ = "f"

    @parse_and_compile
    def g(x):
        return f1(x) + f2(x)

    assert g(5) == 25, g(5)


def test_cast():
    @parse_and_compile
    def f(x):
        return float(x) + 1.5

    assert f(2) == 3.5, f(2)


def test_grad_with_dout():
    """Test grad with dout."""

    def f(x, y):
        a = x ** 3
        b = y ** 4
        return a * b

    @parse_and_compile
    def g(x, y):
        return grad(f, "x")(x, y, dout=2)

    @parse_and_compile
    def h(x, y):
        return grad(f, "y")(x, y, dout=-3)

    x, y = 2.0, 3.0
    dx = 3 * (x ** 2) * (y ** 4)
    dy = 4 * (y ** 3) * (x ** 3)
    _assert_match(dx * 2, g(x, y))
    _assert_match(dy * -3, h(x, y))


def test_for_loop_on_tuple():
    @parse_and_compile
    def f(t):
        ret = 0
        for x in t:
            ret = ret + x
        return ret

    t = (1, 2, 3)
    assert f(t) == 6


def test_operators():
    @parse_and_compile
    def f(a, b):
        return (
            a + b,
            a - b,
            a * b,
            a % b,
            a ** b,
            a == b,
            a != b,
            a < b,
            a > b,
            a <= b,
            a >= b,
            +a,
            -a,
            not a,
            a & b,
            a | b,
            b << a,
            b >> a,
            ~a,
            a and b,
            a or b,
        )

    a = 2
    b = 3
    results = (
        5,
        -1,
        6,
        2,
        8,
        False,
        True,
        True,
        False,
        True,
        False,
        2,
        -2,
        False,
        2,
        3,
        12,
        0,
        ~2,
        3,
        True,
    )
    assert f(a, b) == results, (f(a, b), results)
