from myia.testing.common import Bool, Nil, Ty
from myia.testing.multitest import infer, mt
from myia.testing.testing_inferrers import add_testing_inferrers

add_testing_inferrers()


@mt(
    infer(int, int, result=int),
)
def test_bitwise_and(a, b):
    return a & b


@mt(
    infer(int, int, result=int),
)
def test_bitwise_or(a, b):
    return a | b


@mt(
    infer(int, int, result=int),
)
def test_bitwise_xor(a, b):
    return a ^ b


@mt(
    infer(int, int, result=int),
)
def test_bitwise_lshift(a, b):
    return a << b


@mt(
    infer(int, int, result=int),
)
def test_bitwise_rshift(a, b):
    return a >> b


@mt(
    infer(int, result=int),
)
def test_bitwise_not(a):
    return ~a


@mt(
    # we could not cast to a Nil,
    infer(Ty(Nil), int, result=Exception("wrong number of arguments")),
    infer(Ty(Bool), int, result=Bool),
    infer(Ty(int), float, result=int),
    infer(Ty(int), int, result=int),
)
def test_infer_scalar_cast(dtype, value):
    return dtype(value)
