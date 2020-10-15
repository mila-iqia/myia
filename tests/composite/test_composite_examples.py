import numpy as np

from myia.utils.errors import MyiaTypeError
from myia.xtype import f64, i32, i64

from ..common import Ty, af16_of, af64_of, ai16_of, ai32_of, au64_of
from ..multitest import infer, mt, run
from .examples.operations import composite_full, composite_simple


def Shp(*values):
    """Convert values to a tuple of numpy unsigned integers."""
    return tuple(np.uint64(value) for value in values)


@mt(
    # An error should be raised if wrong values are given as types.
    infer(Shp(2, 3), i32, "bad string", result=MyiaTypeError),
    infer(Shp(2, 3), i32, 10, result=MyiaTypeError),
    infer(Shp(2, 3), i32, (), result=MyiaTypeError),
    # Otherwise, output type should be specified d-type.
    infer(Shp(2, 3), i64, Ty(np.int16), result=ai16_of(2, 3)),
    infer(Shp(2, 3), i64, Ty(np.float16), result=af16_of(2, 3)),
    infer(Shp(2, 3), i64, Ty(np.float64), result=af64_of(2, 3)),
    infer(Shp(2, 3), f64, Ty(np.int16), result=ai16_of(2, 3)),
    infer(Shp(2, 3), f64, Ty(np.int32), result=ai32_of(2, 3)),
    infer(Shp(2, 3), f64, Ty(np.uint64), result=au64_of(2, 3)),
)
def test_infer_composite_full(shape, value, dtype):
    return composite_full(shape, value, dtype)


@mt(
    run(Shp(2, 3), 0, np.float64, result=np.zeros((2, 3))),
    run(Shp(8), 1, np.float16, result=np.ones((8,), "float16")),
    run(Shp(1, 4), -2.5, np.float64, result=(-2.5 * np.ones((1, 4)))),
    run(Shp(1, 4), -2.5, np.double, result=(-2.5 * np.ones((1, 4), "double"))),
    broad_specs=(False, False, False),
)
def test_composite_full(shape, fill_value, dtype):
    return composite_full(shape, fill_value, dtype)


@mt(
    infer(i32, result=i32), infer(f64, result=f64), infer(i64, result=i64),
)
def test_infer_composite_simple(x):
    return composite_simple(x)


@mt(
    run(1, result=((1 + 2) // (3 - 1))),
    run(10, result=-1),
    run(10.0, result=((10 + 2) / (3 - 10))),
)
def test_composite_simple(x):
    return composite_simple(x)
