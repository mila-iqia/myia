import operator

import pytest

from myia.abstract.map import MapError
from myia.abstract.to_abstract import to_abstract, type_to_abstract
from myia.infer.infnode import infer_graph, inferrers, signature
from myia.parser import parse
from myia.utils.info import enable_debug


def fact(x):
    if x <= 1:
        return x
    else:
        return x * fact(x - 1)


def f(x, y):
    # Note: currently / has type (int, float) -> float
    return g(x) / g(y)


def g(z):
    return -z


def test_fact():
    with enable_debug():
        graph = parse(fact)

    g, result = infer_graph(graph, (to_abstract(1),))
    assert result is type_to_abstract(int)

    # TODO: verify that all of g's nodes have a type

    with pytest.raises(MapError):
        infer_graph(graph, (to_abstract(1.5),))


def test_specialization():
    # TODO: This line seems to break str() on representations in other tests,
    # to reproduce, call signature with these arguments before all tests are run
    inferrers[operator.truediv] = signature(int, float, ret=float)

    with enable_debug():
        graph = parse(f)

    g, result = infer_graph(graph, (to_abstract(1), to_abstract(1.5)))
    assert result is type_to_abstract(float)

    # TODO: verify that all of g's nodes have a type
