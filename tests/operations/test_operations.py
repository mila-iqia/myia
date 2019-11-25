import numpy as np

from myia import myia
from myia.xtype import i32, i64, u32

from ..multitest import infer, mt, run, run_no_relay


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run_no_relay(5, 7, result=5)
)
def test_bitwise_and(a, b):
    return a & b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run_no_relay(5, 2, result=7)
)
def test_bitwise_or(a, b):
    return a | b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run_no_relay(10, 8, result=2)
)
def test_bitwise_xor(a, b):
    return a ^ b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(3, 2, result=12)
)
def test_bitwise_lshift(a, b):
    return a << b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(12, 2, result=3)
)
def test_bitwise_rshift(a, b):
    return a >> b


@mt(
    run(np.arange(10), result=np.prod(np.arange(10))),
    run(np.asarray([-2, -1, 2, 11], dtype='float32'), result=np.float32(44)),
)
def test_prod(arr):
    return np.prod(arr)


def test_op_full():
    @myia
    def run_full(value):
        return np.full((2, 7), value, 'float16')

    expected = np.ones((2, 7), 'float16') * 4
    output = run_full(4)
    assert output.dtype == np.float16
    assert np.allclose(expected, output)


def test_op_prod():
    @myia
    def run_prod(val):
        return np.prod(val)

    values = np.arange(1, 6)
    expected = np.prod(values)
    output = run_prod(values)
    assert expected == output == 120
