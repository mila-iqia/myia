import pytest

from types import SimpleNamespace
import numpy as np
import math

from myia.pipeline import scalar_debug_pipeline
from myia.prim.py_implementations import setattr as myia_setattr, \
    tuple_setitem, list_setitem, \
    shape, reshape, array_map, array_scan, array_reduce, \
    distribute, dot, partial as myia_partial, identity, _assert_scalar, \
    switch, scalar_to_array, broadcast_shape, scalar_cast, list_reduce, \
    list_map, env_getitem, env_setitem, env_add, embed, \
    array_to_scalar, transpose, return_, make_record, list_getitem, \
    array_getitem, array_setitem, bool_eq, dict_getitem, tuple_getitem
from myia.utils import newenv

from ..test_lang import parse_compare
from ..common import i64, f16, to_abstract_test, Point, AA


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


@parse_compare(-0.1, 0.3)
def test_prim_tanh(x):
    return math.tanh(x)


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
def test_prim_tuple_getitem(data, item):
    return tuple_getitem(data, item)


@parse_compare(([1, 2, 3], 0), ([4, -6, 7], 2))
def test_prim_list_getitem(data, item):
    return list_getitem(data, item)


@parse_compare((np.array([1, 2, 3]), 0), (np.array([4, -6, 7]), 2))
def test_prim_array_getitem(data, item):
    return array_getitem(data, item)


def test_prim_bool_eq():
    assert bool_eq(False, False)
    assert not bool_eq(False, True)


def test_prim_tuple_setitem():
    tup = (1, 2, 3, 4)
    assert tuple_setitem(tup, 1, 22) == (1, 22, 3, 4)


def test_prim_list_setitem():
    L = [1, 2, 3, 4]
    L2 = [1, 22, 3, 4]
    assert list_setitem(L, 1, 22) == L2
    assert L != L2  # test that this is not inplace


def test_prim_array_setitem():
    L = np.array([1, 2, 3, 4])
    L2 = np.array([1, 22, 3, 4])
    assert np.all(array_setitem(L, 1, 22) == L2)
    assert not np.all(L == L2)  # test that this is not inplace


def test_prim_setattr():
    ns = SimpleNamespace(a=1, b=2)
    ns2 = SimpleNamespace(a=1, b=22)
    assert myia_setattr(ns, 'b', 22) == ns2
    assert ns != ns2  # test that this is not inplace


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


@parse_compare([1, 2, 3])
def test_prim_list_reduce(l):
    def add(a, b):
        return a + b

    return list_reduce(add, l, 4)


@parse_compare([1, 2, 3])
def test_prim_list_map(l):
    def add2(a):
        return a + 2
    return list_map(add2, l)


def test_prim_dict_getitem():
    assert dict_getitem({'x': 2}, 'x') == 2


def test_prim_distribute():
    assert (distribute(1, (2, 3)) == np.ones((2, 3))).all()


def test_prim_reshape():
    assert reshape(np.empty((2, 3)), (6,)).shape == (6,)


def test_prim_transpose():
    assert transpose(np.empty((2, 3)), (1, 0)).shape == (3, 2)
    assert transpose(np.empty((2, 3, 4)), (2, 0, 1)).shape == (4, 2, 3)


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
        assert return_(x) is x


def test_prim_make_record():
    assert make_record(Point, 1, 2) == Point(1, 2)


def test_prim_switch():
    assert switch(True, 1, 2) == 1
    assert switch(False, 1, 2) == 2


def test_scalar_to_array():
    a = scalar_to_array(1, AA)
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.int64
    b = scalar_to_array(1.5, AA)
    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float64


def test_array_to_scalar():
    a = array_to_scalar(np.array(1))
    assert isinstance(a, int)
    assert a == 1
    b = array_to_scalar(np.array(1.5))
    assert isinstance(b, float)
    assert b == 1.5


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
    assert isinstance(scalar_cast(1.5, i64), np.int64)
    assert isinstance(scalar_cast(1.5, f16), np.float16)


def test_env():
    pip1 = scalar_debug_pipeline.select('parse', 'export')
    pip2 = scalar_debug_pipeline

    def f(x, y):
        e1 = env_setitem(newenv, embed(x), 100)

        e2 = env_setitem(newenv, embed(x), 10)
        e2 = env_setitem(e2, embed(y), 20)

        e3 = env_add(e1, e2)

        a = env_getitem(e3, embed(x), 0)
        b = env_getitem(e3, embed(y), 0)
        c = env_getitem(e3, embed(a), 0)

        return (a, b, c)

    res = pip1.run(input=f)['output'](3, 4)
    assert res == (110, 20, 0)

    res = pip2.run(input=f,
                   argspec=(to_abstract_test(i64),
                            to_abstract_test(i64)))['output'](3, 4)
    assert res == (110, 20, 0)
