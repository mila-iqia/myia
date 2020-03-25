import math
from math import (
    cos as math_cos,
    exp as math_exp,
    log as math_log,
    sin as math_sin,
    tan as math_tan,
    tanh as math_tanh,
    trunc as math_trunc,
)

import numpy as np
import pytest

from myia.abstract import ANYTHING, type_to_abstract
from myia.operations import (
    array_cast,
    array_getitem,
    array_map,
    array_reduce,
    array_scan,
    array_setitem,
    array_to_scalar,
    bool_eq,
    broadcast_shape,
    concat,
    distribute,
    dot,
    embed,
    env_add,
    env_getitem,
    env_setitem,
    full,
    identity,
    partial as myia_partial,
    random_initialize,
    random_uint32,
    reshape,
    return_,
    scalar_abs,
    scalar_bit_and,
    scalar_bit_lshift,
    scalar_bit_not,
    scalar_bit_or,
    scalar_bit_rshift,
    scalar_bit_xor,
    scalar_cast,
    scalar_max,
    scalar_sign,
    scalar_to_array,
    shape,
    split,
    stop_gradient,
    switch,
    take,
    take_grad_inp,
    transpose,
    tuple_getitem,
    tuple_setitem,
    unsafe_static_cast,
)
from myia.pipeline import scalar_debug_pipeline
from myia.utils import assert_scalar, newenv

from ..common import AA, f16, i64, to_abstract_test
from ..multitest import mt, run_debug
from ..test_lang import run_lang


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_add(x, y):
    return x + y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_sub(x, y):
    return x - y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_mul(x, y):
    return x * y


@mt(
    run_debug(2.0, 7.0),
    run_debug(4.0, -6.0),
    run_debug(-11, 2),
)
def test_prim_truediv(x, y):
    return x / y


@mt(
    run_debug(2, 7),
    run_debug(4, -6),
    run_debug(-11, 2),
    run_debug(-11.0, 2.0),
    run_debug(0, -1),
)
def test_prim_floordiv(x, y):
    return x // y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_mod(x, y):
    return x % y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_pow(x, y):
    return x ** y


@mt(
    run_debug(-2),
    run_debug(2.3),
    run_debug(-0.6),
)
def test_prim_floor(x):
    return math.floor(x)


@mt(
    run_lang(2, 7),
    run_lang(4, -6.0),
    run_lang(0, -1),
    run_lang(-3.2, 0.0),
)
def test_prim_max(x, y):
    return scalar_max(x, y)


@mt(
    run_lang(-2),
    run_lang(2.3),
    run_lang(-0.6),
)
def test_prim_trunc(x):
    return math_trunc(x)


@mt(
    run_lang(2),
    run_lang(-6),
)
def test_prim_uadd(x):
    return +x


@mt(
    run_lang(2),
    run_lang(-6),
)
def test_prim_usub(x):
    return -x


@mt(
    run_lang(13),
    run_lang(0),
    run_lang(-3),
)
def test_prim_exp(x):
    return math_exp(x)


@mt(
    run_lang(13),
    run_lang(1),
)
def test_prim_log(x):
    return math_log(x)


@mt(
    run_lang(13),
    run_lang(-3),
)
def test_prim_sin(x):
    return math_sin(x)


@mt(
    run_lang(13),
    run_lang(-3),
)
def test_prim_cos(x):
    return math_cos(x)


@mt(
    run_lang(13),
    run_lang(-3),
)
def test_prim_tan(x):
    return math_tan(x)


@mt(
    run_lang(-0.1),
    run_lang(0.3),
)
def test_prim_tanh(x):
    return math_tanh(x)


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_eq(x, y):
    return x == y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_lt(x, y):
    return x < y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_gt(x, y):
    return x > y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_ne(x, y):
    return x != y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_le(x, y):
    return x <= y


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_ge(x, y):
    return x >= y


@mt(
    run_lang(True),
    run_lang(False),
)
def test_prim_not_(x):
    return not x


@mt(
    run_lang(2, 7),
    run_lang(4, -6),
)
def test_prim_tuple(x, y):
    return x, y


@mt(
    run_lang((1, 2, 3), 0),
    run_lang((4, -6, 7), 2),
)
def test_prim_tuple_getitem(data, item):
    return tuple_getitem(data, item)


def test_prim_array_getitem():
    assert array_getitem(np.array([1, 2, 3]), (0,), (1,), (1,)) == [1]
    assert array_getitem(np.array([4, -6, 7]), (2,), (3,), (1,)) == [7]


def test_prim_bool_eq():
    assert bool_eq(False, False)
    assert not bool_eq(False, True)


def test_prim_tuple_setitem():
    tup = (1, 2, 3, 4)
    assert tuple_setitem(tup, 1, 22) == (1, 22, 3, 4)


def test_prim_array_setitem():
    L = np.array([1, 2, 3, 4])
    L2 = np.array([1, 22, 3, 4])
    assert np.all(array_setitem(L, (1,), (2,), (1,), 22) == L2)
    assert not np.all(L == L2)  # test that this is not inplace


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


def test_op_full():
    ref = np.full((4, 9), -4, 'float16')
    res = full((4, 9), -4, np.float16)
    assert res.dtype == 'float16'
    assert np.allclose(ref, res, rtol=0, atol=0)


def test_prim_take():
    inp = np.random.randint(0, 3, (2, 7))
    wgh = np.random.randn(3, 2)
    ref = np.take(wgh, inp, axis=0)
    res = take(wgh, inp)
    assert np.allclose(ref, res, rtol=0, atol=0)


def test_prim_take_grad_inp():
    indices = np.random.randint(0, 3, (2, 7))
    weights = np.random.randn(3, 2)
    dout = take(weights, indices)
    broadcastable_indices = indices.reshape(tuple(indices.shape) + (1,))
    ref = np.zeros(weights.shape, dtype=dout.dtype)
    for i in range(weights.shape[0]):
        ref[i] = (((broadcastable_indices == i) * dout)
                  .reshape((-1, weights.shape[1]))
                  .sum(axis=0))
    res = take_grad_inp(weights.shape[0], indices, dout)
    assert np.allclose(ref, res, rtol=0, atol=0)


def test_prim_scalar_bit_lshift():
    ref = 3 << 2
    res = scalar_bit_lshift(3, 2)
    assert ref == res == 12


def test_prim_scalar_bit_rshift():
    ref = 12 >> 2
    res = scalar_bit_rshift(12, 2)
    assert ref == res == 3


def test_prim_scalar_bit_and():
    ref = 5 & 7
    res = scalar_bit_and(5, 7)
    assert ref == res == 5


def test_prim_scalar_bit_or():
    ref = 5 | 2
    res = scalar_bit_or(5, 2)
    assert ref == res == 7


def test_prim_scalar_bit_xor():
    ref = 10 ^ 8
    res = scalar_bit_xor(10, 8)
    assert ref == res == 2


def test_prim_scalar_bit_not():
    a = np.uint16(0b0000110101001101)
    ref = np.uint16(0b1111001010110010)
    assert ref == ~a
    res = scalar_bit_not(a)
    assert res == ref


@run_lang(40)
def test_prim_partial(x):
    def f(a, b):
        return a + b

    g = myia_partial(f, 2)
    return g(x)


def test_assert_scalar():
    assert_scalar(0)
    assert_scalar(1.0, 2.0)
    assert_scalar(np.ones(()))
    # with pytest.raises(TypeError):
    #     assert_scalar(1, 1.0)
    with pytest.raises(TypeError):
        assert_scalar(np.ones((2, 2)))
    with pytest.raises(TypeError):
        assert_scalar((1, 2), (3, 4))


def test_prim_identity():
    for x in (1, 1.7, True, False, [1, 2, 3], (4, 5)):
        assert identity(x) is x
        assert return_(x) is x


def test_prim_stop_gradient():
    for x in (1, 1.7, True, False, [1, 2, 3], (4, 5)):
        assert stop_gradient(x) is x


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
        ((2, ANYTHING, 4), (ANYTHING, 3, ANYTHING), (2, 3, 4)),
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
    assert isinstance(scalar_cast(1.5, type_to_abstract(i64)), np.int64)
    assert isinstance(scalar_cast(1.5, type_to_abstract(f16)), np.float16)


def test_array_cast():
    assert isinstance(array_cast(np.array([1.5, 1.7]),
                                 type_to_abstract(i64)), np.ndarray)
    assert (array_cast(np.array([1.5, 1.7]),
                       type_to_abstract(i64))).dtype == np.dtype(np.int64)
    assert isinstance(array_cast(np.array([1.5, 1.7]),
                                 type_to_abstract(f16)), np.ndarray)
    assert (array_cast(np.array([1.5, 1.]),
                       type_to_abstract(f16))).dtype == np.dtype(np.float16)


def test_prim_concat():
    a = np.array([[1.5, 1.7]])
    b = np.array([[2.3, 0.5]])

    ref = np.concatenate((a, b), 1)
    res = concat((a, b), 1)

    assert (res == ref).all()


def test_prim_split():
    a = np.array([0., 1., 2., 3.])

    ref = np.split(a, (1, 2), 0)
    res = split(a, (1, 1, 2), 0)

    for _ref, _res in zip(ref, res):
        assert (_res == _ref).all()


@mt(
    run_lang(7),
    run_lang(-6),
    run_lang(0),
)
def test_prim_scalar_abs(x):
    return scalar_abs(x)


@mt(
    run_lang(7),
    run_lang(-6),
    run_lang(0),
)
def test_prim_scalar_sign(x):
    return scalar_sign(x)


def test_prim_unsafe_static_cast():
    assert unsafe_static_cast(1234, type_to_abstract(float)) == 1234


def test_env():

    def f(x, y):
        e1 = env_setitem(newenv, embed(x), 100)

        e2 = env_setitem(newenv, embed(x), 10)
        e2 = env_setitem(e2, embed(y), 20)

        e3 = env_add(e1, e2)

        a = env_getitem(e3, embed(x), 0)
        b = env_getitem(e3, embed(y), 0)
        c = env_getitem(e3, embed(a), 0)

        return (a, b, c)

    res = scalar_debug_pipeline.run(
        input=f,
        argspec=(to_abstract_test(i64),
                 to_abstract_test(i64))
    )['output'](3, 4)
    assert res == (110, 20, 0)


def test_call_operation():
    from myia.operations import Operation, scalar_pow
    assert scalar_pow(2, 10) == 1024

    bad_op = Operation('bad_op')
    with pytest.raises(RuntimeError):
        bad_op(123)


def test_random():
    rstate = random_initialize(1234)
    r0, v0 = random_uint32(rstate, (2, 3))
    r1, v1 = random_uint32(rstate, (2, 3))
    assert isinstance(rstate, np.random.RandomState)
    assert isinstance(r0, np.random.RandomState)
    assert isinstance(r1, np.random.RandomState)
    assert isinstance(v0, np.ndarray)
    assert isinstance(v1, np.ndarray)
    assert v0.dtype == 'uint32'
    assert v1.dtype == 'uint32'
    assert v0.shape == (2, 3)
    assert v1.shape == (2, 3)
    assert not np.allclose(v0, v1)
