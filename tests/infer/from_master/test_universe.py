from myia.abstract.map import MapError as InferenceError
from myia.basics import (
    global_universe_getitem as universe_getitem,
    global_universe_setitem as universe_setitem,
    make_handle,
)
from myia.testing.common import H
from myia.testing.multitest import infer, mt

typeof = type


@mt(
    infer(int, result=H(int)),
    infer((int, float), result=H((int, float))),
)
def test_make_handle(x):
    return make_handle(typeof(x))


@mt(
    infer(H(int), result=int),
    infer(int, result=InferenceError),
)
def test_universe_getitem(h):
    return universe_getitem(h)


@mt(
    infer(H(int), int, result=None),
    infer(H(int), float, result=InferenceError),
    infer(int, int, result=InferenceError),
)
def test_universe_setitem(h, v):
    return universe_setitem(h, v)


@mt(
    infer(H(int), int, int, result=None),
    infer(H(int), float, float, result=InferenceError),
)
def test_universe(h, x, y):
    init = universe_getitem(h)
    universe_setitem(h, x + y)
    xy = universe_getitem(h)
    return universe_setitem(h, init * xy)
