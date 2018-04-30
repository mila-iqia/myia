import pytest

from myia.dtype import Bool, Float, Function, Int, List, Number, Struct, \
    Tuple, Type, TypeMeta, UInt

from myia.unify import Unification, var


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


def test_visit():
    U = Unification()
    v1 = var()
    v2 = var()

    l1 = List(v1)
    l2 = U.clone(l1)
    assert l1 is not l2
    assert U.unify(l1, l2)[v1] is l2.element_type

    s1 = Struct(a=v1, b=v2)
    s2 = U.clone(s1)
    assert s1 is not s2
    assert set(s2.elements.keys()) == {'a', 'b'}

    t1 = Tuple(v1, v1, v2)
    t2 = U.clone(t1)
    assert t1 is not t2
    assert t2.elements[0] is t2.elements[1]
    assert t2.elements[0] is not t2.elements[2]
    assert len(t2.elements) == 3

    c1 = Function((v1, v2), v2)
    c2 = U.clone(c1)
    assert c1 is not c2
    assert c2.arguments[1] is c2.retval
    assert c2.arguments[0] is not c2.arguments[1]
    assert len(c2.arguments) == 2

    b = U.clone(Bool())
    assert b is Bool()


def test_Number():
    assert isinstance(Int(32), Number)
    assert not isinstance(Bool(), Number)
    with pytest.raises(RuntimeError):
        Number()
    with pytest.raises(ValueError):
        Float(33)


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
