import math

import numpy as np

from myia import operations
from myia.testing.common import (
    MA,
    Ty,
    af16_of,
    af32_of,
    af64_of,
    ai16_of,
    ai32_of,
    ai64_of,
    au64_of,
)
from myia.testing.multitest import infer, mt, run
from myia.utils.errors import MyiaTypeError
from myia.xtype import Bool, Nil, f16, f32, f64, i8, i16, i32, i64, u32, u64


def Shp(*values):
    """Convert values to a tuple of numpy unsigned itnegers."""
    return tuple(np.uint64(value) for value in values)


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(5, 7, result=5),
)
def test_bitwise_and(a, b):
    return a & b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(5, 2, result=7),
)
def test_bitwise_or(a, b):
    return a | b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(10, 8, result=2),
)
def test_bitwise_xor(a, b):
    return a ^ b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(3, 2, result=12),
)
def test_bitwise_lshift(a, b):
    return a << b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
    run(12, 2, result=3),
)
def test_bitwise_rshift(a, b):
    return a >> b


@mt(
    infer(u32, result=u32),
    infer(i32, result=i32),
    infer(i64, result=i64),
    run(0, result=~0),
    run(-37, result=~(-37)),
    run(np.uint16(0b0000110101001101), result=np.uint16(0b1111001010110010)),
)
def test_bitwise_not(a):
    return ~a


@mt(
    # seems to not accept integers
    run(0.0, result=0),
    run(np.float16(0), result=0),
    run(np.float32(0), result=0),
    run(np.float64(0), result=0),
    run(math.pi / 2, result=1),
    run(2 * math.pi / 3, result=math.sin(2 * math.pi / 3)),
)
def test_sin(a):
    return math.sin(a)


@mt(
    # seems to not accept integers
    run(0.0, result=1),
    run(np.float16(0), result=1),
    run(np.float32(0), result=1),
    run(np.float64(0), result=1),
    run(math.pi / 2, result=0),
    run(2 * math.pi / 3, result=math.cos(2 * math.pi / 3)),
)
def test_cos(a):
    return math.cos(a)


@mt(
    # seems to not accept integers
    run(0.0, result=0),
    run(np.float16(0), result=0),
    run(np.float32(0), result=0),
    run(np.float64(0), result=0),
    run(math.pi / 4, result=1),
    run(2 * math.pi / 3, result=math.tan(2 * math.pi / 3)),
)
def test_tan(a):
    return math.tan(a)


@mt(
    run(-2.7, result=-2),
    run(7.8, result=7),
    run(np.float16(7.8), result=7),
    run(np.float32(7.8), result=7),
    run(np.float64(7.8), result=7),
)
def test_trunc(a):
    return math.trunc(a)


@mt(
    # sin seems not supported for pytorch/CPU/float16
    run(MA(3, 3, dtype="float32"), result=np.sin(MA(3, 3, dtype="float32"))),
    run(MA(3, 3, dtype="float64"), result=np.sin(MA(3, 3, dtype="float64"))),
)
def test_elemwise_sin(a):
    return np.sin(a)


@mt(
    # cos seems not supported for pytorch/CPU/float16
    run(MA(3, 3, dtype="float32"), result=np.cos(MA(3, 3, dtype="float32"))),
    run(MA(3, 3, dtype="float64"), result=np.cos(MA(3, 3, dtype="float64"))),
)
def test_elemwise_cos(a):
    return np.cos(a)


@mt(
    # cos seems not supported for pytorch/CPU/float16
    run(MA(3, 3, dtype="float32"), result=np.tan(MA(3, 3, dtype="float32"))),
    run(MA(3, 3, dtype="float64"), result=np.tan(MA(3, 3, dtype="float64"))),
)
def test_elemwise_tan(a):
    return np.tan(a)


@mt(
    run(MA(3, 3, dtype="float32"), result=np.trunc(MA(3, 3, dtype="float32"))),
    run(MA(3, 3, dtype="float64"), result=np.trunc(MA(3, 3, dtype="float64"))),
)
def test_elemwise_trunc(a):
    return np.trunc(a)


@mt(
    infer(af16_of(2, 3), result=af16_of()),
    infer(ai32_of(2, 3), result=ai32_of()),
    infer(au64_of(2, 3), result=au64_of()),
    run(np.arange(1, 11), result=3628800),
    run(np.asarray([-2, -1, 2, 11], dtype="float32"), result=np.float32(44)),
)
def test_prod(arr):
    return np.prod(arr)


@mt(
    run(Shp(2, 3), 0, None, result=np.zeros((2, 3))),
    run(Shp(8), 1, "float16", result=np.ones((8,), "float16")),
    run(Shp(1, 4), -2.5, None, result=(-2.5 * np.ones((1, 4)))),
    run(Shp(1, 4), -2.5, "double", result=(-2.5 * np.ones((1, 4), "double"))),
    broad_specs=(False, False, False),
)
def test_full(shape, value, dtype):
    return np.full(shape, value, dtype)


@mt(
    # An error should be raised if wrong values are given as types.
    infer(Shp(2, 3), i32, "bad string", result=MyiaTypeError),
    infer(Shp(2, 3), i32, 10, result=MyiaTypeError),
    infer(Shp(2, 3), i32, (), result=MyiaTypeError),
    # If d-type is not specified, output type should be type of fill value.
    infer(Shp(2, 3), i16, None, result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, None, result=ai64_of(2, 3)),
    infer(Shp(2, 3), f16, None, result=af16_of(2, 3)),
    infer(Shp(2, 3), f32, None, result=af32_of(2, 3)),
    # Otherwise, output type should be specified d-type.
    infer(Shp(2, 3), i64, "int16", result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, "float16", result=af16_of(2, 3)),
    infer(Shp(2, 3), i64, "float64", result=af64_of(2, 3)),
    infer(Shp(2, 3), f64, "int16", result=ai16_of(2, 3)),
    infer(Shp(2, 3), f64, "int32", result=ai32_of(2, 3)),
    infer(Shp(2, 3), f64, "uint64", result=au64_of(2, 3)),
    # Numpy d-types should also be accepted as d-types.
    infer(Shp(2, 3), i64, Ty(np.int16), result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, Ty(np.float16), result=af16_of(2, 3)),
    infer(Shp(2, 3), f64, Ty(np.uint64), result=au64_of(2, 3)),
)
def test_infer_full(shape, value, dtype):
    return np.full(shape, value, dtype)


@mt(
    # we could not cast to a Nil,
    infer(Ty(Nil), i64, result=MyiaTypeError),
    # We could cast to a Bool,
    infer(Ty(Bool), i64, result=Bool),
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
    broad_specs=(False, False),
)
def test_scalar_cast(dtype, value):
    return dtype(value)


@mt(
    run((7, 4, 9), 0, result=7),
    run((7, 4, 9), 1, result=4),
    run((7, 4, 9), 2, result=9),
    # Only index must be passed as raw (constant) value
    broad_specs=(True, False),
)
def test_tuple_getitem(t, i):
    return t[i]


@mt(
    run((7, 4, 9), 0, 44, result=(44, 4, 9)),
    run((7, 4, 9), 1, 44, result=(7, 44, 9)),
    run((7, 4, 9), 2, 44, result=(7, 4, 44)),
    # Only index must be passed as raw value
    broad_specs=(True, False, True),
)
def test_tuple_setitem(t, i, v):
    return operations.tuple_setitem(t, i, v)
