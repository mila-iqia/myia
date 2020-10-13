from myia.operations import composite_simple
from myia.xtype import f64, i32, i64

from ..multitest import infer, mt, run


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
