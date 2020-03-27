from myia.lib import InferenceError
from myia.operations import dtype, scalar_cast

from ..common import MA, Ty, af32_of, f32, i64, to_abstract_test
from ..multitest import infer, mt, run


@mt(
    infer(i64, result=InferenceError),
    infer(af32_of(4, 5), result=Ty(to_abstract_test(f32))),
)
def test_dtype(arr):
    return dtype(arr)


@mt(infer(af32_of(4, 5), i64, result=f32), run(MA(2, 3), 7, result=7.0))
def test_cast_to_dtype(arr, x):
    return scalar_cast(x, dtype(arr))
