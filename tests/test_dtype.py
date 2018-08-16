import pytest
import numpy
from dataclasses import dataclass

from myia.dtype import Bool, Float, Function, Int, List, Number, \
    Tuple, Type, TypeMeta, UInt, np_dtype_to_type, type_to_np_dtype, Object, \
    pytype_to_myiatype, Array, Class, External


@dataclass(frozen=True)
class Point:
    x: Number
    y: Number

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


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


def test_Object():
    with pytest.raises(RuntimeError):
        Object()


def test_cache():
    f32 = Float(32)
    assert f32 is Float(32)
    assert f32 is not Float(16)
    assert f32 is not Float(64)


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


def test_type_conversions():
    assert np_dtype_to_type('float32') is Float(32)

    with pytest.raises(TypeError):
        np_dtype_to_type('float80')

    assert type_to_np_dtype(Float(16)) == 'float16'

    with pytest.raises(TypeError):
        type_to_np_dtype(List(Int(64)))


def test_pytype_to_myiatype():
    ptm = pytype_to_myiatype

    assert ptm(Int) is Int
    assert ptm(Int(64)) is Int(64)

    assert ptm(bool) is Bool()
    assert ptm(int) is Int(64)
    assert ptm(float) is Float(64)

    assert ptm(tuple) is Tuple
    assert ptm(tuple, (1, (2, 3))) is Tuple(Int(64), Tuple(Int(64), Int(64)))

    assert ptm(list) is List
    assert ptm(list, [1, 2, 3]) is List(Int(64))
    with pytest.raises(TypeError):
        ptm(list, [1, (2, 3)])

    assert ptm(numpy.ndarray) is Array
    assert ptm(numpy.ndarray, numpy.ones((2, 2))) is Array(Float(64))

    # We run this before ptm(Point) to make sure it doesn't cache Float, Float
    # for the type of x and y
    ptm(Point, Point(1.1, 2.2))

    pcls = ptm(Point)
    assert isinstance(pcls, Class)
    assert pcls.attributes == {'x': Number, 'y': Number}
    assert 'abs' in pcls.methods
    assert str(pcls.tag) == 'Point'

    pcls2 = ptm(Point, Point(1, 2))
    assert pcls2.tag is pcls.tag
    assert pcls2.attributes == {'x': Int(64), 'y': Int(64)}

    assert ptm(str) is External(str)
    assert ptm(object) is External(object)
