
import pytest

from myia import dtype as ty, dshape as sh
from myia.infer import ANYTHING, MyiaTypeError
from myia.prim.py_implementations import typeof
from myia.abstract.base import (
    AbstractScalar as _S, AbstractTuple as T, AbstractArray as A,
    AbstractList as L, AbstractClass as C,
    shapeof,
    amerge,
    Possibilities as _Poss
)


def S(v=ANYTHING, t=None, s=None):
    return _S({
        'value': v,
        'type': t or typeof(v),
        'shape': s or (type_to_shape(t) if v is ANYTHING else shapeof(v))
    })


def Poss(*things):
    return _S({
        'value': _Poss(things),
        'type': typeof(things[0]),
        'shape': shapeof(things[0])
    })


def type_to_shape(typ):
    """Default value for ShapeTrack."""
    if ty.ismyiatype(typ, ty.Array):
        raise Exception(
            'There is no default value for Arrays on the shape track.'
        )  # pragma: no cover
    if ty.ismyiatype(typ, ty.Tuple):
        return sh.TupleShape(type_to_shape(e) for e in typ.elements)
    elif ty.ismyiatype(typ, ty.List):
        return sh.ListShape(type_to_shape(typ.element_type))
    elif ty.ismyiatype(typ, ty.Class):
        return sh.ClassShape(dict((attr, type_to_shape(tp))
                                for attr, tp in typ.attributes.items()))
    return sh.NOSHAPE


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
