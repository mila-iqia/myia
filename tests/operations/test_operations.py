import math

import numpy as np

from myia.testing.common import (
    Ty,
    af16_of,
    af32_of,
    af64_of,
    ai16_of,
    ai32_of,
    ai64_of,
    au64_of,
)
from myia.testing.multitest import infer, mt
from myia.utils.errors import MyiaTypeError
from myia.xtype import Bool, Nil, f16, f32, f64, i8, i16, i32, i64, u32, u64


def Shp(*values):
    """Convert values to a tuple of numpy unsigned integers."""
    return tuple(np.uint64(value) for value in values)


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
)
def test_bitwise_and(a, b):
    return a & b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
)
def test_bitwise_or(a, b):
    return a | b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
)
def test_bitwise_xor(a, b):
    return a ^ b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
)
def test_bitwise_lshift(a, b):
    return a << b


@mt(
    infer(u32, u32, result=u32),
    infer(i32, i32, result=i32),
    infer(i64, i64, result=i64),
)
def test_bitwise_rshift(a, b):
    return a >> b


@mt(
    infer(u32, result=u32),
    infer(i32, result=i32),
    infer(i64, result=i64),
)
def test_bitwise_not(a):
    return ~a


@mt(
    infer(af16_of(2, 3), result=af16_of()),
    infer(ai32_of(2, 3), result=ai32_of()),
    infer(au64_of(2, 3), result=au64_of()),
)
def test_prod(arr):
    return np.prod(arr)


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
