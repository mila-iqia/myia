import operator

from myia.abstract.map import MapError
from myia.abstract.to_abstract import to_abstract, type_to_abstract
from myia.infer.infnode import infer_graph, inferrers, signature
from myia.parser import parse
from myia.testing.common import Nil, Ty
from myia.testing.multitest import infer, mt
from myia.utils.info import enable_debug


def f(x, y):
    # Note: test_specialization overrides / to have type (int, float) -> float
    return g(x) / g(y)


def g(z):
    return -z


def test_specialization():
    # TODO: This line seems to break str() on representations in other tests,
    # to reproduce, call signature with these arguments before all tests are run
    inferrers[operator.truediv] = signature(int, float, ret=float)

    with enable_debug():
        graph = parse(f)

    g = infer_graph(graph, (to_abstract(1), to_abstract(1.5)))
    result = g.return_.abstract
    assert result is type_to_abstract(float)

    # TODO: verify that all of g's nodes have a type


@mt(
    infer(int, int, result=int),
    infer(float, float, result=float),
    infer(int, float, result=MapError),
    infer(float, int, result=Exception(".*Cannot merge.*")),
)
def test_sum(a, b):
    return a + b


# Test `infer` as a decorator
@infer(int, int, result=int)
def test_sum_2(a, b):
    return a + b


@mt(
    infer(int, int, result=int),
    infer(float, float, result=float),
    infer(int, float, result=MapError),
    # This is for coverage of missing builtin function, it could be
    # a different builtin when we add support for list
    infer(list, list, result=TypeError),
)
def test_sum_method(a, b):
    return a.__add__(b)


def fact(x):
    if x <= 1:
        return x
    else:
        return x * fact(x - 1)


@mt(
    infer(int, result=int),
    infer(float, result=MapError),
)
def test_fact(x):
    return fact(x)


@infer(int, result=TypeError)
def test_not_function(x):
    return x()


@infer(result=int)
def test_nullary_call():
    def f():
        return 1

    return f()


@infer(int, result=int)
def test_constant_branch(x):
    if x <= 0:
        return 1
    else:
        return 2


@infer(int, result=int)
def test_module_function_call(x):
    return operator.neg(x)


@mt(
    # we could not cast to a Nil,
    infer(Ty(Nil), int, result=Exception("wrong number of arguments")),
    infer(Ty(bool), bool, result=bool),
    infer(Ty(bool), int, result=bool),
    infer(Ty(bool), float, result=bool),
    infer(Ty(int), bool, result=int),
    infer(Ty(int), int, result=int),
    infer(Ty(int), float, result=int),
    infer(Ty(float), bool, result=float),
    infer(Ty(float), int, result=float),
    infer(Ty(float), float, result=float),
)
def test_infer_scalar_cast(dtype, value):
    return dtype(value)
