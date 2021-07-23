from myia.operations import (
    make_handle,
    typeof,
    universe_getitem,
    universe_setitem,
)
from myia.testing.common import H, f64, i64
from myia.testing.multitest import infer, mt
from myia.utils import HandleInstance, InferenceError, new_universe
from myia.xtype import EnvType, UniverseType


@mt(
    infer(UniverseType, i64, result=H(i64)),
    infer(UniverseType, (i64, f64), result=H((i64, f64))),
)
def test_make_handle(U, x):
    return make_handle(typeof(x), U)[1]


@mt(
    infer(UniverseType, H(i64), result=i64),
    infer(EnvType, H(i64), result=InferenceError),
    infer(UniverseType, i64, result=InferenceError),
)
def test_universe_getitem(U, h):
    return universe_getitem(U, h)


@mt(
    infer(UniverseType, H(i64), i64, result=UniverseType),
    infer(UniverseType, H(i64), f64, result=InferenceError),
    infer(EnvType, H(i64), i64, result=InferenceError),
    infer(UniverseType, i64, i64, result=InferenceError),
)
def test_universe_setitem(U, h, v):
    return universe_setitem(U, h, v)


def _test_universe_chk(args, U):
    U0, h, x, y = args
    assert U.get(h) == h.state * (x + y)
    return True


@mt(
    infer(UniverseType, H(i64), i64, i64, result=UniverseType),
    infer(UniverseType, H(i64), f64, f64, result=InferenceError),
)
def test_universe(U, h, x, y):
    init = universe_getitem(U, h)
    U = universe_setitem(U, h, x + y)
    xy = universe_getitem(U, h)
    return universe_setitem(U, h, init * xy)


def test_universe_commit():
    U = new_universe
    h1 = HandleInstance(2)
    h2 = HandleInstance(7)
    U = U.set(h1, 20)
    U = U.set(h2, 70)
    assert h1.state == 2
    assert h2.state == 7
    U.commit()
    assert h1.state == 20
    assert h2.state == 70
