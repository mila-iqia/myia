
import pytest

from myia import dtype as ty
from myia.prim.py_implementations import typeof
from myia.abstract import (
    ANYTHING, MyiaTypeError,
    AbstractScalar as _S, AbstractTuple as T, AbstractArray as A,
    AbstractList as L, AbstractClass as C,
    amerge,
    Possibilities as _Poss,
    VALUE, TYPE
)


def S(v=ANYTHING, t=None, s=None):
    return _S({
        VALUE: v,
        TYPE: t or typeof(v),
    })


def Poss(*things):
    return _S({
        VALUE: _Poss(things),
        TYPE: typeof(things[0]),
    })


def test_merge():
    a = T([S(1), S(t=ty.Int[64])])
    b = T([S(1), S(t=ty.Int[64])])
    c = T([S(t=ty.Int[64]), S(t=ty.Int[64])])

    assert amerge(a, b, loop=None, forced=False) is a
    assert amerge(a, c, loop=None, forced=False) == c
    assert amerge(c, a, loop=None, forced=False) is c

    with pytest.raises(MyiaTypeError):
        amerge(a, c, loop=None, forced=True)


def test_merge_possibilities():
    a = Poss(1, 2)
    b = Poss(2, 3)
    c = Poss(2)
    assert amerge(a, b,
                  loop=None,
                  forced=False) == Poss(1, 2, 3)
    assert amerge(a, c,
                  loop=None,
                  forced=False) is a

    with pytest.raises(MyiaTypeError):
        amerge(a, b, loop=None, forced=True)

    assert amerge(a, c, loop=None, forced=True) is a
