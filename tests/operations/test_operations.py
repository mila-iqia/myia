import numpy as np

from myia.utils.errors import MyiaTypeError
from myia.xtype import f16, f32, f64, i16, i32, i64, u32

from ..common import (
    Ty,
    af16_of,
    af32_of,
    af64_of,
    ai16_of,
    ai32_of,
    ai64_of,
    au64_of,
)
from ..multitest import infer, mt, run, run_no_relay


def Shp(*values):
    """Convert values to a tuple of numpy unsigned itnegers."""
    return tuple(np.uint64(value) for value in values)


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
    infer(af16_of(2, 3), result=af16_of()),
    infer(ai32_of(2, 3), result=ai32_of()),
    infer(au64_of(2, 3), result=au64_of()),
    run(np.arange(1, 11), result=3628800),
    run(np.asarray([-2, -1, 2, 11], dtype='float32'), result=np.float32(44)),
)
def test_prod(arr):
    return np.prod(arr)


@mt(
    run(Shp(2, 3), 0, None, result=np.zeros((2, 3))),
    run(Shp(8,), 1, 'float16', result=np.ones((8,), 'float16')),
    run(Shp(1, 4), -2.5, None, result=(-2.5 * np.ones((1, 4)))),
    run(Shp(1, 4), -2.5, 'double', result=(-2.5 * np.ones((1, 4), 'double'))),
    broad_specs=(False, False, False),
)
def test_full(shape, value, dtype):
    return np.full(shape, value, dtype)


@mt(
    # An error should be raised if wrong values are given as types.
    infer(Shp(2, 3), i32, 'bad string', result=MyiaTypeError),
    infer(Shp(2, 3), i32, 10, result=MyiaTypeError),
    infer(Shp(2, 3), i32, (), result=MyiaTypeError),
    # If d-type is not specified, output type should be type of fill value.
    infer(Shp(2, 3), i16, None, result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, None, result=ai64_of(2, 3)),
    infer(Shp(2, 3), f16, None, result=af16_of(2, 3)),
    infer(Shp(2, 3), f32, None, result=af32_of(2, 3)),
    # Otherwise, output type should be specified d-type.
    infer(Shp(2, 3), i64, 'int16', result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, 'float16', result=af16_of(2, 3)),
    infer(Shp(2, 3), i64, 'float64', result=af64_of(2, 3)),
    infer(Shp(2, 3), f64, 'int16', result=ai16_of(2, 3)),
    infer(Shp(2, 3), f64, 'int32', result=ai32_of(2, 3)),
    infer(Shp(2, 3), f64, 'uint64', result=au64_of(2, 3)),
    # Numpy d-types should also be accepted as d-types.
    infer(Shp(2, 3), i64, Ty(np.int16), result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, Ty(np.float16), result=af16_of(2, 3)),
    infer(Shp(2, 3), f64, Ty(np.uint64), result=au64_of(2, 3)),
)
def test_infer_full(shape, value, dtype):
    return np.full(shape, value, dtype)
