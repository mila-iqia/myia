import numpy as np

from myia.pipeline import scalar_debug_compile as compile
from myia.composite import list_reduce
from myia.prim.py_implementations import \
    array_map, array_reduce, array_scan, scalar_usub, list_map
from myia.utils import list_to_cons

from .test_lang import parse_compare


@parse_compare(([1, 2, 3], [4, 5, 6]))
def test_vm_icall_fn(l1, l2):
    def mulm(x, y):
        return x * -y

    return list_map(mulm, l1, l2)


@parse_compare(([1, 2, 3],))
def test_vm_icall_prim(l):
    return list_map(scalar_usub, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_clos(l):
    y = 1 + 1

    def add2(v):
        return v + y

    return list_map(add2, l)


@parse_compare((2, 3), (2.0, 3.0))
def test_vm_floordiv(x, y):
    return x // y


@parse_compare((2, 3), (2.0, 3.0))
def test_vm_truediv(x, y):
    return x / y


def test_vm_array_map():
    @compile
    def f(x):
        def add1(x):
            return x + 1

        return array_map(add1, x)

    a = np.zeros((2, 3))
    res = f(a)
    assert (res == np.ones((2, 3))).all()


def test_vm_array_map2():
    @compile
    def f(xs, ys):
        def add(x, y):
            return x + y

        return array_map(add, xs, ys)

    a = np.ones((2, 3))
    b = np.ones((2, 3))
    res = f(a, b)
    assert (res == 2 * np.ones((2, 3))).all()


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

        return array_reduce(add, x, (1, 3))

    a = np.ones((2, 3))
    res = f(a)
    assert (res == a.sum(axis=0)).all()


def test_vm_list_reduce():
    @compile
    def f(x):
        def add(x, y):
            return x + y

        return list_reduce(add, x, 4)

    a = list_to_cons([1, 2, 3])
    res = f(a)
    assert res == 10
