from myia.prim.py_implementations import maplist, usub

from .test_lang import parse_compare


@parse_compare(([1, 2, 3],))
def test_vm_icall_fn(l):
    def square(x):
        return x * x

    return maplist(square, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_prim(l):
    return maplist(usub, l)


@parse_compare(([1, 2, 3],))
def test_vm_icall_clos(l):
    y = 1 + 1

    def add2(v):
        return v + y

    return maplist(add2, l)
