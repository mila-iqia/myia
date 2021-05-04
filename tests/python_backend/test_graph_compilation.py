import pytest

from myia.compile.backends.python.python import compile_graph
from myia.parser import parse
from myia.utils.info import enable_debug


def parse_and_compile(function):
    with enable_debug():
        graph = parse(function)
    return compile_graph(graph, debug=True)


# NB: Need to be global for test_recursion to work.
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


# NB: Need to be global for test_function_call to work.
def global_pgcd(a, b):
    return a if not b else global_pgcd(b, a % b)


def test_simple():
    @parse_and_compile
    def f(x):
        return 2 * x + 1

    assert f(0) == 1, f(0)
    assert f(-2) == -3, f(0)


def test_cast():
    @parse_and_compile
    def f(x):
        return float(x) + 1.5

    assert f(2) == 3.5, f(2)


def test_operators():
    @parse_and_compile
    def f(a, b):
        t = (1, 2, 4)
        return (
            a + b,
            a - b,
            a / b,
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
            a ^ b,
            b << a,
            b >> a,
            ~a,
            a and b,
            a or b,
            a in t,
            b in t,
            b // a,
            t[2],
            False is True,
            False is False,
            False is not True,
            False is not False,
        )

    _a = 2
    _b = 3
    results = (
        5,
        -1,
        2 / 3,
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
        1,
        12,
        0,
        ~2,
        3,
        True,
        True,
        False,
        1,
        4,
        False,
        True,
        True,
        False,
    )
    assert f(_a, _b) == results, (f(_a, _b), results)


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


def test_for_loop_on_tuple():
    @parse_and_compile
    def f(t):
        ret = 0
        for x in t:
            ret = ret + x
        return ret

    tpl = (1, 2, 3)
    assert f(tpl) == 6


def test_while_if():
    @parse_and_compile
    def f(n):
        ret = 0
        i = 0
        while i <= n:
            if i % 2 == 0:
                ret = ret + i
            else:
                ret = ret + 0.1
            i = i + 1
        return ret

    assert f(0) == 0
    assert f(1) == 0.1
    assert f(2) == 2.1
    assert f(3) == 2.2
    assert f(4) == 6.2
    assert f(5) == 6.3
    assert f(6) == 12.3
    assert f(7) == 12.4
    assert f(8) == 20.4
    assert f(9) == 20.5
    assert f(10) == 30.5


def test_if_while():
    @parse_and_compile
    def f(n):
        ret = 0
        if n % 2 == 0:
            i = 0
            while i <= n:
                ret = ret + i
                i = i + 1
        else:
            i = n
            while i >= 0:
                ret = ret + 2 * i
                i = i - 1
        return ret

    assert f(0) == 0
    assert f(1) == 2
    assert f(2) == 3
    assert f(3) == 12
    assert f(4) == 10
    assert f(5) == 30


def test_recursion():
    f = parse_and_compile(factorial)

    for x, y in (
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
    ):
        assert f(x) == y, f(x)


def test_function_call():
    @parse_and_compile
    def pgcd(a, b):
        return global_pgcd(b, a) if a < b else global_pgcd(a, b)

    assert pgcd(12, 32) == 4, pgcd(12, 32)
    assert pgcd(625, 250) == 125, pgcd(625, 250)


@pytest.mark.xfail(reason="KeyError generated with universe_getitem()")
def test_inner_closure():
    @parse_and_compile
    def pgcd(a, b):
        def local_pgcd(_a, _b):
            return _a if not _b else local_pgcd(_b, _a % _b)

        return local_pgcd(b, a) if a < b else local_pgcd(a, b)

    assert pgcd(12, 32) == 4, pgcd(12, 32)
    assert pgcd(625, 250) == 125, pgcd(625, 250)


def test_deep_closure():
    @parse_and_compile
    def f(x):
        def f0(x):
            return x // 3

        def f1(x):
            def f2(x):
                def f3(x):
                    return 2 * f0(x)

                return f3(x) - 1

            return f2(x) + x

        return f1(x)

    assert f(100) == 165
