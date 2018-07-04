import pytest

from types import SimpleNamespace
import numpy as np

from myia.dtype import Int, Float, List, Tuple, External
from myia.prim.py_implementations import head, setattr as myia_setattr, \
    setitem as myia_setitem, tail, hastype, typeof, \
    shape, reshape, map_array, scan_array, reduce_array, distribute, dot, \
    partial as myia_partial, _assert_scalar

from ..test_lang import parse_compare


@parse_compare((2, 7), (4, -6))
def test_prim_add(x, y):
    return x + y


@parse_compare((2, 7), (4, -6))
def test_prim_sub(x, y):
    return x - y


@parse_compare((2, 7), (4, -6))
def test_prim_mul(x, y):
    return x * y


@parse_compare((2, 7), (4, -6))
def test_prim_div(x, y):
    return x / y


@parse_compare((2, 7), (4, -6))
def test_prim_mod(x, y):
    return x % y


@parse_compare((2, 7), (4, -6))
def test_prim_pow(x, y):
    return x ** y


@parse_compare(2, -6)
def test_prim_uadd(x):
    return +x


@parse_compare(2, -6)
def test_prim_usub(x):
    return -x


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


def test_prim_head_tail():
    tup = (1, 2, 3, 4)
    assert head(tup) == 1
    assert tail(tup) == (2, 3, 4)


def test_prim_setitem():
    tup = (1, 2, 3, 4)
    assert myia_setitem(tup, 1, 22) == (1, 22, 3, 4)

    L = [1, 2, 3, 4]
    L2 = [1, 22, 3, 4]
    assert myia_setitem(L, 1, 22) == L2
    assert L != L2  # test that this is not inplace


def test_prim_setattr():
    ns = SimpleNamespace(a=1, b=2)
    ns2 = SimpleNamespace(a=1, b=22)
    assert myia_setattr(ns, 'b', 22) == ns2
    assert ns != ns2  # test that this is not inplace


def test_prim_typeof():
    i64 = Int(64)
    f64 = Float(64)
    assert typeof(1) == i64
    assert typeof(1.2) == f64
    assert typeof((1, 2.0, (3, 4))) == Tuple(i64, f64, Tuple(i64, i64))
    assert typeof([1, 2]) == List(i64)
    with pytest.raises(TypeError):
        typeof([1, 2, 3.4])
    assert typeof(object()) == External(object)


def test_prim_istype():
    i64 = Int(64)
    f64 = Float(64)
    assert hastype(123, i64)
    assert not hastype(123.4, i64)
    assert hastype(123.4, f64)
    assert hastype([1, 2, 3], List)
    assert hastype([1, 2, 3], List(i64))
    assert hastype((1, 2.0, (3, 4)), Tuple)
    assert hastype((1, 2.0, (3, 4)), Tuple(i64, f64, Tuple(i64, i64)))
    with pytest.raises(TypeError):
        hastype([1, 2, 3.4], List)
    assert hastype(object(), External(object))


def test_prim_shape():
    v = np.empty((2, 3))
    assert shape(v) == (2, 3)


def test_prim_map_array():
    v = np.zeros((2, 3))

    def f(a):
        return a + 1

    v2 = map_array(f, v)

    assert (v == 0).all()
    assert (v2 == 1).all()


def test_prim_scan_array():
    v = np.ones((2, 3))

    def f(a, b):
        return a + b

    vref = np.cumsum(v, axis=1)
    v2 = scan_array(f, 0, v, 1)

    assert (v == 1).all()
    assert (v2 == vref).all()


def test_prim_reduce_array():
    v = np.ones((2, 3))

    def f(a, b):
        return a + b

    vref = np.add.reduce(v, axis=1)
    v2 = reduce_array(f, 0, v, 1)

    assert (v == 1).all()
    assert (v2 == vref).all()


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
