import asyncio
import pytest
import numpy
from dataclasses import dataclass

from myia.dtype import ismyiatype, Bool, Float, Function, Int, List, Number, \
    Tuple, UInt, SymbolicKeyType, np_dtype_to_type, type_to_np_dtype, Object, \
    pytype_to_myiatype as ptm, Array, Class, External, JTagged, get_generic, \
    type_cloner, type_cloner_async
from myia.utils import SymbolicKeyInstance


@dataclass(frozen=True)
class Point:
    x: Number
    y: Number

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_instantiate():
    with pytest.raises(RuntimeError):
        Object()
    with pytest.raises(RuntimeError):
        Int[64]()


def test_cache():
    f32 = Float[32]
    assert f32 is Float[32]
    assert f32 is not Float[16]
    assert f32 is not Float[64]


def test_Number():
    assert ismyiatype(Int[32], Number)
    assert not ismyiatype(Bool, Number)
    with pytest.raises(ValueError):
        Float[33]


def test_List():
    ll = List[Bool]
    assert ll.element_type is Bool


def test_Tuple():
    t = Tuple[Float[64], Int[8], Bool]
    assert t.elements[1] is Int[8]
    t1 = Tuple[(Bool, Int[8])]
    t2 = Tuple[[Bool, Int[8]]]
    assert t1 == t2
    assert t1 is t2
    t3 = Tuple[Bool, Int[8]]
    assert t1 is t3
    assert Tuple[Float[16]] is Tuple[(Float[16],)]


def test_Function():
    c = Function[(), Float[32]]
    assert c.retval is Float[32]
    c2 = Function[[], Float[32]]
    assert c is c2


def test_make_subtype():
    with pytest.raises(TypeError):
        Bool[64]
    with pytest.raises(RuntimeError):
        Number[64]
    with pytest.raises(TypeError):
        Int[64][64]
    with pytest.raises(TypeError):
        Bool.make_subtype(nbits=64)
    with pytest.raises(TypeError):
        Int.make_subtype(x=64)
    with pytest.raises(TypeError):
        Int.make_subtype()


def test_generic():
    assert Number.generic is Number
    assert Int.generic is Int
    assert Int[64].generic is Int


def test_is_generic():
    assert Number.is_generic()
    assert Int.is_generic()
    assert not Int[64].is_generic()


def test_ismyiatype():
    assert ismyiatype(Number)
    assert ismyiatype(Number, Object)
    assert ismyiatype(Int)
    assert ismyiatype(Int, Number)
    assert ismyiatype(Int[64])
    assert ismyiatype(Int[64], Int)
    assert ismyiatype(Int[64], Number)
    assert ismyiatype(Int[64], Object)
    assert not ismyiatype(Int[64], Float)
    assert ismyiatype(Tuple[()], Tuple)
    assert ismyiatype(Int, generic=True)
    assert not ismyiatype(Int, generic=False)
    assert not ismyiatype(Int[64], generic=True)
    assert ismyiatype(Int[64], generic=False)
    assert not ismyiatype(1234)
    assert not ismyiatype(object)


def test_get_generic():
    assert get_generic(Int[64], Int[32]) is Int
    assert get_generic(Tuple[Number, Number], Tuple[()]) is Tuple
    assert get_generic(Int[64], Float[64]) is None


def test_repr():
    t = UInt[16]
    assert repr(t) == 'UInt[16]'


def test_type_conversions():
    assert np_dtype_to_type('float32') is Float[32]

    with pytest.raises(TypeError):
        np_dtype_to_type('float80')

    assert type_to_np_dtype(Float[16]) == 'float16'

    with pytest.raises(TypeError):
        type_to_np_dtype(List[Int[64]])


def test_pytype_to_myiatype():
    assert ptm(Int) is Int
    assert ptm(Int[64]) is Int[64]

    assert ptm(bool) is Bool
    assert ptm(int) is Int[64]
    assert ptm(float) is Float[64]

    assert ptm(tuple) is Tuple
    assert ptm(tuple, (1, (2, 3))) is Tuple[Int[64], Tuple[Int[64], Int[64]]]

    assert ptm(list) is List
    assert ptm(list, [1, 2, 3]) is List[Int[64]]
    with pytest.raises(TypeError):
        ptm(list, [1, (2, 3)])

    assert ptm(numpy.ndarray) is Array
    assert ptm(numpy.ndarray, numpy.ones((2, 2))) is Array[Float[64]]

    assert ptm(SymbolicKeyInstance) is SymbolicKeyType

    # We run this before ptm(Point) to make sure it doesn't cache Float, Float
    # for the type of x and y
    ptm(Point, Point(1.1, 2.2))

    pcls = ptm(Point)
    assert issubclass(pcls, Class)
    assert pcls.attributes == {'x': Number, 'y': Number}
    assert 'abs' in pcls.methods
    assert str(pcls.tag) == 'Point'

    pcls2 = ptm(Point, Point(1, 2))
    assert pcls2.tag is pcls.tag
    assert pcls2.attributes == {'x': Int[64], 'y': Int[64]}

    assert ptm(str) is External[str]
    assert ptm(object) is External[object]


def test_type_cloner():

    t1 = Function[
        [List[JTagged[Int[64]]], List[Float[32]]],
        Tuple[Array[Int[32]], Class['C', {'x': Int[64]}, {}]]
    ]

    t2 = Function[
        [List[JTagged[Int[32]]], List[Float[32]]],
        Tuple[Array[Int[16]], Class['C', {'x': Int[32]}, {}]]
    ]

    assert type_cloner(t1) is t1

    @type_cloner.variant
    def halver(self, t: Int):
        return Int[t.bits // 2]

    assert halver(t1) is t2

    assert asyncio.run(type_cloner_async(t1)) is t1

    @type_cloner_async.variant
    async def halver_async(self, t: Int):
        return Int[t.bits // 2]

    assert asyncio.run(halver_async(t1)) is t2
