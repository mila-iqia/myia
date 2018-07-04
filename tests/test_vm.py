import numpy as np

from myia.api import compile
from myia.prim.py_implementations import (map_array, maplist, reduce_array,
                                          scan_array, scalar_usub)

from .test_lang import parse_compare


@parse_compare(([1, 2, 3],))
def test_vm_icall_fn(l):
    def square(x):
        return x * x

    return maplist(square, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_prim(l):
    return maplist(scalar_usub, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_clos(l):
    y = 1 + 1

    def add2(v):
        return v + y

    return maplist(add2, l)


def test_vm_map_array():
    @compile
    def f(x):
        def add1(x):
            return x + 1

        return map_array(add1, x)

    a = np.zeros((2, 3))
    res = f(a)
    assert (res == np.ones((2, 3))).all()


def test_vm_scan_array():
    @compile
    def f(x):
        def add(x, y):
            return x + y

        return scan_array(add, 0, x, 1)

    a = np.ones((2, 3))
    res = f(a)
    assert (res == a.cumsum(axis=1)).all()


def test_vm_reduce_array():
    @compile
    def f(x):
        def add(x, y):
            return x + y

        return reduce_array(add, 0, x, 0)

    a = np.ones((2, 3))
    res = f(a)
    assert (res == a.sum(axis=0)).all()
