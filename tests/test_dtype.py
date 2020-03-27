import pytest

from myia.utils import EnvInstance, SymbolicKeyInstance
from myia.xtype import (
    Bool,
    EnvType,
    Float,
    Int,
    Number,
    Object,
    SymbolicKeyType,
    UInt,
    np_dtype_to_type,
    pytype_to_myiatype as ptm,
    type_to_np_dtype,
)


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
    assert issubclass(Int[32], Number)
    assert not issubclass(Bool, Number)
    with pytest.raises(ValueError):
        Float[33]


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


def test_repr():
    t = UInt[16]
    assert repr(t) == "UInt[16]"


def test_type_conversions():
    assert np_dtype_to_type("float32") is Float[32]

    with pytest.raises(TypeError):
        np_dtype_to_type("float80")

    assert type_to_np_dtype(Float[16]) == "float16"

    with pytest.raises(TypeError):
        type_to_np_dtype(Object)


def test_pytype_to_myiatype():
    assert ptm(bool) is Bool
    assert ptm(int) is Int[64]
    assert ptm(float) is Float[64]
    assert ptm(SymbolicKeyInstance) is SymbolicKeyType
    assert ptm(EnvInstance) is EnvType
