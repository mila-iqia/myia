import pytest

from types import SimpleNamespace

from myia.dtype import Int, Float, List, Tuple
from myia.prim.py_implementations import head, setattr as myia_setattr, \
    setitem as myia_setitem, tail, hastype, typeof

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


@parse_compare((2, 7), (4, -6))
def test_prim_not_(x, y):
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
    with pytest.raises(TypeError):
        typeof(object())


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
    with pytest.raises(TypeError):
        hastype(object(), i64)
