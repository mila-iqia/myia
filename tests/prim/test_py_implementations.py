import pytest

from types import SimpleNamespace
import numpy as np
import math

from myia.dtype import Int, Float, List, Tuple, Class, Number, External
from myia.prim.py_implementations import setattr as myia_setattr, \
    tuple_setitem, list_setitem, tail, hastype, typeof, \
    shape, reshape, array_map, array_scan, array_reduce, \
    distribute, dot, partial as myia_partial, identity, _assert_scalar, \
    switch, scalar_to_array, broadcast_shape, scalar_cast, list_reduce, \
    issubtype

from ..test_lang import parse_compare
from ..common import i64, f64


@parse_compare((2, 7), (4, -6))
def test_prim_add(x, y):
    return x + y


@parse_compare((2, 7), (4, -6))
def test_prim_sub(x, y):
    return x - y


@parse_compare((2, 7), (4, -6))
def test_prim_mul(x, y):
    return x * y


@parse_compare((2.0, 7.0), (4.0, -6.0), (-11, 2))
def test_prim_truediv(x, y):
    return x / y


@parse_compare((2, 7), (4, -6), (-11, 2), (-11.0, 2.0), (0, -1))
def test_prim_floordiv(x, y):
    return x // y


@parse_compare((2, 7), (4, -6))
def test_prim_mod(x, y):
    return x % y


@parse_compare((2, 7), (4, -6))
def test_prim_pow(x, y):
    return x ** y


@parse_compare(-2, 2.3, -0.6)
def test_prim_floor(x):
    return math.floor(x)


@parse_compare(-2, 2.3, -0.6)
def test_prim_trunc(x):
    return math.trunc(x)


@parse_compare(2, -6)
def test_prim_uadd(x):
    return +x


@parse_compare(2, -6)
def test_prim_usub(x):
    return -x


@parse_compare(13, 0, -3)
def test_prim_exp(x):
    return math.exp(x)


@parse_compare(13, 1)
def test_prim_log(x):
    return math.log(x)


@parse_compare(13, -3)
def test_prim_sin(x):
    return math.sin(x)


@parse_compare(13, -3)
def test_prim_cos(x):
    return math.cos(x)


@parse_compare(13, -3)
def test_prim_tan(x):
    return math.tan(x)


@parse_compare((2, 7), (4, -6))
def test_prim_eq(x, y):
    return x == y


@parse_compare((2, 7), (4, -6))
def test_prim_lt(x, y):
    return x < y


@parse_compare((2, 7), (4, -6))
def test_prim_gt(x, y):
    return x > y


@parse_compare((2, 7), (4, -6))
def test_prim_ne(x, y):
    return x != y


@parse_compare((2, 7), (4, -6))
def test_prim_le(x, y):
    return x <= y


@parse_compare((2, 7), (4, -6))
def test_prim_ge(x, y):
    return x >= y


@parse_compare((True,), (False,))
def test_prim_not_(x):
    return not x


@parse_compare((2, 7), (4, -6))
def test_prim_tuple(x, y):
    return x, y


@parse_compare(((1, 2, 3), 0), ((4, -6, 7), 2))
def test_prim_getitem(data, item):
    return data[item]


def test_prim_tail():
    tup = (1, 2, 3, 4)
    assert tail(tup) == (2, 3, 4)


def test_prim_tuple_setitem():
    tup = (1, 2, 3, 4)
    assert tuple_setitem(tup, 1, 22) == (1, 22, 3, 4)


def test_prim_list_setitem():
    L = [1, 2, 3, 4]
    L2 = [1, 22, 3, 4]
    assert list_setitem(L, 1, 22) == L2
    assert L != L2  # test that this is not inplace


def test_prim_setattr():
    ns = SimpleNamespace(a=1, b=2)
    ns2 = SimpleNamespace(a=1, b=22)
    assert myia_setattr(ns, 'b', 22) == ns2
    assert ns != ns2  # test that this is not inplace


def test_prim_typeof():
    assert typeof(1) == i64
    assert typeof(1.2) == f64
    assert typeof((1, 2.0, (3, 4))) == Tuple[i64, f64, Tuple[i64, i64]]
    assert typeof([1, 2]) == List[i64]
    with pytest.raises(TypeError):
        typeof([1, 2, 3.4])
    assert typeof(object()) == External[object]


def test_issubtype():
    assert issubtype(Tuple[i64, i64], Tuple)
    assert issubtype(Tuple[i64, i64], Tuple[i64, i64])
    assert issubtype(Tuple[i64, i64], Tuple[Number, Number])
    assert not issubtype(Tuple[i64, i64], Tuple[i64, i64, i64])

    AN = Class["A", {'x': Number}, {}]
    Ai = Class["A", {'x': i64}, {}]
    Bi = Class["B", {'x': i64}, {}]
    assert issubtype(AN, Class)
    assert issubtype(Ai, AN)
    assert not issubtype(AN, Ai)
    assert not issubtype(Bi, AN)


def test_prim_ismyiatype():
    i64 = Int[64]
    f64 = Float[64]
    assert hastype(123, i64)
    assert not hastype(123.4, i64)
    assert hastype(123.4, f64)
    assert hastype([1, 2, 3], List)
    assert hastype([1, 2, 3], List[i64])
    assert hastype((1, 2.0, (3, 4)), Tuple)
    assert hastype((1, 2.0, (3, 4)), Tuple[i64, f64, Tuple[i64, i64]])
    with pytest.raises(TypeError):
        hastype([1, 2, 3.4], List)
    assert hastype(object(), External[object])


def test_prim_shape():
    v = np.empty((2, 3))
    assert shape(v) == (2, 3)


def test_prim_array_map():
    v = np.zeros((2, 3))

    def f(a):
        return a + 1

    v2 = array_map(f, v)

    assert (v == 0).all()
    assert (v2 == 1).all()


def test_prim_array_map2():
    v1 = np.ones((2, 3))
    v2 = np.ones((2, 3))

    def f(a, b):
        return a + b

    vres = array_map(f, v1, v2)

    assert (v1 == 1).all()
    assert (v2 == 1).all()
    assert (vres == 2).all()


def test_prim_array_scan():
    v = np.ones((2, 3))

    def f(a, b):
        return a + b

    vref = np.cumsum(v, axis=1)
    v2 = array_scan(f, 0, v, 1)

    assert (v == 1).all()
    assert (v2 == vref).all()


def test_prim_array_reduce():
    def add(a, b):
        return a + b

    tests = [
        (add, (2, 3, 7), (1, 3, 1), 14),
        (add, (2, 3, 7), (1, 3, 8), ValueError),
        (add, (2, 3, 7), (1, 2, 3, 7), ValueError),
        (add, (2, 3, 7), (3, 1), 14),
        (add, (2, 3, 7), (1, 1, 1), 42),
        (add, (2, 3, 7), (), 42),
    ]

    for f, inshp, outshp, value in tests:
        v = np.ones(inshp)
        try:
            res = array_reduce(f, v, outshp)
        except Exception as e:
            if isinstance(value, type) and isinstance(e, value):
                continue
            else:
                print(f'Expected {value}, but got {e}')
                raise

        assert res.shape == outshp
        assert (res == value).all()


def test_prim_list_reduce():
    def add(a, b):
        return a + b

    assert list_reduce(add, [1, 2, 3], 4) == 10


def test_prim_distribute():
    assert (distribute(1, (2, 3)) == np.ones((2, 3))).all()


def test_prim_reshape():
    assert reshape(np.empty((2, 3)), (6,)).shape == (6,)


def test_prim_dot():
    a = np.ones((2, 3))
    b = np.ones((3, 4))

    ref = np.dot(a, b)
    res = dot(a, b)

    assert (res == ref).all()


@parse_compare((40,),)
def test_prim_partial(x):
    def f(a, b):
        return a + b

    g = myia_partial(f, 2)
    return g(x)


def test_assert_scalar():
    _assert_scalar(0)
    _assert_scalar(1.0, 2.0)
    _assert_scalar(np.ones(()))
    # with pytest.raises(TypeError):
    #     _assert_scalar(1, 1.0)
    with pytest.raises(TypeError):
        _assert_scalar(np.ones((2, 2)))
    with pytest.raises(TypeError):
        _assert_scalar((1, 2), (3, 4))


def test_prim_identity():
    for x in (1, 1.7, True, False, [1, 2, 3], (4, 5)):
        assert identity(x) is x


def test_prim_switch():
    assert switch(True, 1, 2) == 1
    assert switch(False, 1, 2) == 2


def test_scalar_to_array():
    a = scalar_to_array(1)
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.int64
    b = scalar_to_array(1.5)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float64


def test_broadcast_shape():
    tests = [
        ((2, 3), (2, 3), (2, 3)),
        ((2, 1), (2, 3), (2, 3)),
        ((2, 3), (2, 1), (2, 3)),
        ((2, 1), (1, 3), (2, 3)),
        ((2, 1), (7, 1, 3), (7, 2, 3)),
        ((1, 2, 3), (2, 3), (1, 2, 3)),
        ((2, 3), (2, 4), ValueError),
        ((), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
        ((1, 2, 3, 4, 5), (), (1, 2, 3, 4, 5)),
    ]
    for shpx, shpy, result in tests:
        try:
            shp = broadcast_shape(shpx, shpy)
        except Exception as e:
            if isinstance(result, type) and isinstance(e, result):
                continue
            else:
                print(f'Expected {result}, got {e}')
                raise
        assert shp == result


def test_scalar_cast():
    assert isinstance(scalar_cast(1.5, Int[64]), np.int64)
    assert isinstance(scalar_cast(1.5, Float[16]), np.float16)
