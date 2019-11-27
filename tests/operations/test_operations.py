import numpy as np

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


@mt(
    run((2, 3), 0, None, result=np.zeros((2, 3), 'float32')),
    run((8,), 1, 'float16', result=np.ones((8,), 'float16')),
    run((1, 4), -2.5, None, result=(-2.5 * np.ones((1, 4), 'float32'))),
    run((1, 4), -2.5, 'float64', result=(-2.5 * np.ones((1, 4), 'float64'))),
    broad_specs=(False, False, False),
)
def test_full(shape, value, dtype):
    return np.full(shape, value, dtype)
