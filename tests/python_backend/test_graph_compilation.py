from myia.compile.backends.python.python import compile_graph
from myia.ir import manage
from myia.ir.utils import print_graph
from myia.parser import parse


def _compile(fn):
    g = parse(fn)
    manage(g)
    print(print_graph(g))
    cf = compile_graph(g, debug=True)
    return cf


def test_while():
    @_compile
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
    @_compile
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


def test_closure_call():
    def _pgcd(a, b):
        return a if not b else _pgcd(b, a % b)

    @_compile
    def pgcd(a, b):
        return _pgcd(b, a) if a < b else _pgcd(a, b)

    assert pgcd(12, 32) == 4, pgcd(12, 32)
    assert pgcd(625, 250) == 125, pgcd(625, 250)


def test_from_parse_with_if():
    @_compile
    def f(x):
        if x < 0:
            return x * x
        else:
            return x * x * x

    assert f(-2) == 4, f(-2)
    assert f(0) == 0, f(0)
    assert f(2) == 8, f(2)


def test_from_parse():
    @_compile
    def f(x):
        return 2 * x + 1

    assert f(0) == 1, f(0)
    assert f(-2) == -3, f(0)
