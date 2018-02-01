
from pytest import mark
from myia.py_implementations import (
    make_tuple,
    getitem as myia_getitem,
    setitem as myia_setitem
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


@parse_compare(([1, 2, 3], 0, 33), ([4, -6, 7], 2, 41))
def test_prim_setitem(data, item, value):
    data[item] = value
    return data[item]
