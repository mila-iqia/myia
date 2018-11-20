import pytest

import math
import numpy as np

from myia.prim.py_implementations import distribute, scalar_to_array, dot

from ..test_compile import parse_compare
from ..common import MA, MB


@parse_compare((2, 3))
def test_add(x, y):
    return x + y


@parse_compare((2, 3))
def test_sub(x, y):
    return x - y


@parse_compare((2, 3))
def test_mul(x, y):
    return x * y


@pytest.mark.xfail(reason="truediv doesn't work for ints")
@parse_compare((2, 3), (2.0, 3.0))
def test_truediv(x, y):
    return x / y


@parse_compare((2, 3), (2.0, 3.0))
def test_floordiv(x, y):
    return x // y


@parse_compare((2, 3))
def test_mod(x, y):
    return x % y


@parse_compare((2.0, 3.0))
def test_pow(x, y):
    return x ** y


@parse_compare((2,))
def test_uadd(x):
    return +x


@parse_compare((2,))
def test_usub(x):
    return -x


@parse_compare((2.0,))
def test_exp(x):
    return math.exp(x)


@parse_compare((2.0,))
def test_log(x):
    return math.log(x)


@pytest.mark.xfail(reason="not implemented")
@parse_compare((2.0,))
def test_tan(x):
    return math.tan(x)


@parse_compare((2, 3))
def test_eq(x, y):
    return x == y


@parse_compare((2, 3))
def test_lt(x, y):
    return x < y


@parse_compare((2, 3))
def test_gt(x, y):
    return x > y


@parse_compare((2, 3))
def test_ne(x, y):
    return x != y


@parse_compare((2, 3))
def test_le(x, y):
    return x <= y


@parse_compare((2, 3))
def test_ge(x, y):
    return x >= y


@parse_compare((2,))
def test_to_array(x):
    return scalar_to_array(x)


@parse_compare((False,), (True,))
def test_bool_not(x,):
    return not x


@parse_compare((2,), array=True)
def test_distribute(x):
    return distribute(scalar_to_array(x), (2, 3))


@parse_compare(np.ones((1, 3)), np.ones((3,)), array=True)
def test_distribute2(x):
    return distribute(x, (2, 3))


@parse_compare(MA(2, 3), array=True, optimize=False)
def test_distribute3(x):
    return distribute(x, (2, 3))


@parse_compare((MA(2, 3), MB(2, 3)),
               (MA(1, 3), MB(2, 3)),
               (MA(2, 1), MB(2, 3)),
               array=True)
def test_array_map(x, y):
    return x + y


@parse_compare((MA(2, 3), MB(3, 4)),
               array=True)
def test_dot(x, y):
    return dot(x, y)
