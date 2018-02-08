
from pytest import mark
from types import SimpleNamespace

from myia.py_implementations import (
    head, tail,
    setitem as myia_setitem,
    setattr as myia_setattr
)


from .test_lang import parse_compare


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

    l = [1, 2, 3, 4]
    l2 = [1, 22, 3, 4]
    assert myia_setitem(l, 1, 22) == l2
    assert l != l2  # test that this is not inplace


def test_prim_setattr():
    ns = SimpleNamespace(a=1, b=2)
    ns2 = SimpleNamespace(a=1, b=22)
    assert myia_setattr(ns, 'b', 22) == ns2
    assert ns != ns2  # test that this is not inplace
