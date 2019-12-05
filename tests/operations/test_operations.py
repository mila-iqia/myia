import numpy as np

from myia.utils.errors import MyiaTypeError
from myia.xtype import f16, f32, f64, i8, i16, i32, i64, u32, u64

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


@mt(
    # we could create an int8 from any floating.
    infer(Ty(np.int8), f16, result=i8),
    infer(Ty(np.int8), f32, result=i8),
    infer(Ty(np.int8), f64, result=i8),
    # we could cast an int64 to any lower precision integer.
    infer(Ty(np.int8), i64, result=i8),
    infer(Ty(np.int16), i64, result=i16),
    infer(Ty(np.int32), i64, result=i32),
    infer(Ty(np.int64), i64, result=i64),
    # we could instantiate an uint from an int, and vice versa
    infer(Ty(np.int64), u64, result=i64),
    infer(Ty(np.uint64), i64, result=u64),
)
def test_infer_scalar_cast(dtype, value):
    return dtype(value)


@mt(
    # test each scalar type
    run(np.uint8, 0, result=0),
    run(np.uint16, 0, result=0),
    run(np.uint32, 0, result=0),
    run(np.uint64, 0, result=0),
    run(np.int8, 0, result=0),
    run(np.int16, 0, result=0),
    run(np.int32, 0, result=0),
    run(np.int64, 0, result=0),
    run(np.float16, 0, result=0),
    run(np.float32, 0, result=0),
    run(np.float64, 0, result=0),
    run(np.bool, 0, result=0),
    run(np.int, 0, result=0),
    run(np.float, 0, result=0),
    run(np.double, 0, result=0),
    run(bool, 0, result=0),
    run(int, 0, result=0),
    run(float, 0, result=0),
    # test bool
    run(np.bool, 0.0, result=False),
    run(np.bool, 1, result=True),
    run(np.bool, 1, result=1),
    run(np.bool, -1, result=1),
    run(np.bool, -1.23456, result=1),
    run(np.bool, -1.23456, result=1),
    # test uint8
    run(np.uint8, 0, result=0),
    run(np.uint8, 255, result=255),
    run(np.uint8, 256, result=0),
    run(np.uint8, 257, result=1),
    run(np.uint8, -1, result=255),
    run(np.uint8, -1.5, result=255),  # -1.5 => -1 => forced to 255
    run(np.uint8, 255.123456789, result=255),
    # test int8
    run(np.int8, -128, result=-128),
    run(np.int8, 127, result=127),
    run(np.int8, 128, result=-128),
    run(np.int8, 129, result=-127),
    run(np.int8, -129, result=127),
    # test float16
    run(np.float16, 1, result=1),
    run(np.float16, -1.1, result=np.float16(-1.1)),
    run(np.float16, -1.23456789, result=-1.234375),
    broad_specs=(False, False)
)
def test_scalar_cast(dtype, value):
    return dtype(value)
