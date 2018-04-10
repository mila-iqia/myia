import pytest

from myia.dtype import (TypeMeta, Type, Bool, Number, Float, UInt, Int,
                        List, Struct, Tuple, Function)


def test_TypeMeta():
    with pytest.raises(TypeError):
        class A(tuple, int, metaclass=TypeMeta):
            pass

    class B(metaclass=TypeMeta):
        pass

    with pytest.raises(TypeError):
        class C(metaclass=TypeMeta):
            __slots__ = ('c',)


def test_Type():
    with pytest.raises(RuntimeError):
        Type()


def test_cache():
    f32 = Float(32)
    assert f32 is Float(32)
    assert f32 is not Float(16)
    assert f32 is not Float(64)


def test_Number():
    assert isinstance(Int(32), Number)
    assert not isinstance(Bool(), Number)


def test_List():
    ll = List(Bool())
    assert ll.element_type is Bool()


def test_Struct():
    c64 = Struct((('r', Float(32)), ('i', Float(32))))
    assert c64.r is Float(32)
    assert c64.elements['i'] is Float(32)

    with pytest.raises(AttributeError):
        c64.foo

    c64_2 = Struct(dict(r=Float(32), i=Float(32)))
    assert c64 is c64_2

    c64_3 = Struct(r=Float(32), i=Float(32))
    assert c64 is c64_3


def test_Tuple():
    t = Tuple((Float(64), Int(8), Bool()))
    assert t.elements[1] is Int(8)
    t1 = Tuple((Bool(), Int(8)))
    t2 = Tuple([Bool(), Int(8)])
    assert t1 == t2
    assert t1 is t2
    t3 = Tuple(Bool(), Int(8))
    assert t1 is t3
    assert Tuple(Float(16)) is Tuple((Float(16),))


def test_Function():
    c = Function((), Float(32))
    assert c.retval is Float(32)
    c2 = Function([], Float(32))
    assert c is c2


def test_repr():
    t = UInt(16)
    assert repr(t) == 'UInt(16)'
