from myia.abstract.map import MapError
from myia.testing.common import A
from myia.testing.multitest import infer, mt


# Test `mt`
@mt(
    infer(A(int), A(int), result=A(int)),
    infer(A(float), A(float), result=A(float)),
    infer(int, int, result=int),
    infer(float, float, result=float),
)
def test_sum(a, b):
    return a + b


# Test `infer` alone
@infer(A(int), A(int), result=A(int))
def test_sum_2(a, b):
    return a + b


@infer(int, float, result=MapError)
def test_sum_3(a, b):
    return a + b


@infer(int, float, result=Exception(".*Cannot merge.*"))
def test_sum_4(a, b):
    return a + b
