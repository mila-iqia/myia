from myia.abstract.map import MapError as InferenceError
from myia.basics import (
    global_universe_getitem as universe_getitem,
    global_universe_setitem as universe_setitem,
    make_handle,
)
from myia.testing.common import H, f64, i64
from myia.testing.multitest import infer, mt

typeof = type


@mt(
    infer(i64, result=H(i64)),
    infer((i64, f64), result=H((i64, f64))),
)
def test_make_handle(x):
    return make_handle(typeof(x))


@mt(
    infer(H(i64), result=i64),
    infer(i64, result=InferenceError),
)
def test_universe_getitem(h):
    return universe_getitem(h)


@mt(
    infer(H(i64), i64, result=None),
    infer(H(i64), f64, result=InferenceError),
    infer(i64, i64, result=InferenceError),
)
def test_universe_setitem(h, v):
    return universe_setitem(h, v)


@mt(
    infer(H(i64), i64, i64, result=None),
    infer(H(i64), f64, f64, result=InferenceError),
)
def test_universe(h, x, y):
    init = universe_getitem(h)
    universe_setitem(h, x + y)
    xy = universe_getitem(h)
    return universe_setitem(h, init * xy)
