"""Test some debug utilities that may be used in tests."""


import pytest

from myia.pipeline import scalar_parse as parse
from myia.debug.utils import GraphIndex


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
