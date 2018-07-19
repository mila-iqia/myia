import numpy as np

from myia.api import compile
from myia.prim.py_implementations import (array_map, list_map, array_reduce,
                                          array_scan, scalar_usub)

from .test_lang import parse_compare


@parse_compare(([1, 2, 3],))
def test_vm_icall_fn(l):
    def square(x):
        return x * x

    return list_map(square, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_prim(l):
    return list_map(scalar_usub, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_clos(l):
    y = 1 + 1

    def add2(v):
        return v + y

    return list_map(add2, l)


def test_vm_array_map():
    @compile
    def f(x):
        def add1(x):
            return x + 1

        return array_map(add1, x)

    a = np.zeros((2, 3))
    res = f(a)
    assert (res == np.ones((2, 3))).all()


def test_vm_array_scan():
    @compile
    def f(x):
        def add(x, y):
            return x + y

        return array_scan(add, 0, x, 1)

    a = np.ones((2, 3))
    res = f(a)
    assert (res == a.cumsum(axis=1)).all()


def test_vm_array_reduce():
    @compile
    def f(x):
        def add(x, y):
            return x + y

        return array_reduce(add, 0, x, 0)

    a = np.ones((2, 3))
    res = f(a)
    assert (res == a.sum(axis=0)).all()
