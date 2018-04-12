"""Test some debug utilities that may be used in tests."""


import pytest
from myia.api import parse
from myia.debug.utils import isomorphic, GraphIndex


def _check_isomorphic(g1, g2, expected=True):
    # Check that it works both ways
    assert isomorphic(g1, g2) == expected
    assert isomorphic(g2, g1) == expected


def test_isomorphic():
    @parse
    def f1(x, y):
        return x * y

    @parse
    def f2(a, b):
        return a * b

    @parse
    def f3(a, b):
        return a + b

    @parse
    def f4(x, y, z):
        return x * y

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, False)
    _check_isomorphic(f1, f4, False)
    _check_isomorphic(f4, f1, False)


def test_isomorphic_closures():
    @parse
    def f1(x):
        def inner1(y):
            return x * y
        return inner1(10)

    @parse
    def f2(a):
        def inner2(b):
            return a * b
        return inner2(10)

    @parse
    def f3(a):
        def inner3(b):
            return a + b
        return inner3(10)

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, False)


def test_isomorphic_globals():
    def helper1(x):
        return x * x

    def helper2(a):
        return a * a

    def helper3(a):
        return a

    @parse
    def f1(x):
        return helper1(x) * helper1(4)

    @parse
    def f2(a):
        return helper1(a) * helper1(4)

    @parse
    def f3(a):
        return helper2(a) * helper1(4)

    @parse
    def f4(a):
        return helper2(a) * helper3(4)

    _check_isomorphic(f1, f2, True)
    _check_isomorphic(f1, f3, True)
    _check_isomorphic(f1, f4, False)


def test_GraphIndex():
    @parse
    def f(x, y):
        a = x * y
        b = x + y
        c = a - b
        return c

    idx = GraphIndex(f)

    assert idx['x'] is f.parameters[0]
    assert idx['y'] is f.parameters[1]

    assert idx['c'] is f.output
    assert idx['a'] is f.output.inputs[1]
    assert idx['b'] is f.output.inputs[2]

    with pytest.raises(Exception):
        idx['d']


def test_GraphIndex_multigraph():

    def helper(x):
        return x * x

    @parse
    def f(x, y):
        def inner(a):
            b = a - 1000
            return b

        a = inner(x) * helper(y)
        return a

    idx = GraphIndex(f)

    assert idx.get_all('x') == {idx['f'].parameters[0],
                                idx['helper'].parameters[0]}

    assert idx.get_all('y') == {idx['f'].parameters[1]}

    assert idx.get_all('a') == {idx['f'].output,
                                idx['inner'].parameters[0]}

    assert idx.get_all('b') == {idx['inner'].output}
