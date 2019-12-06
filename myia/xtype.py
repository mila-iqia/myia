"""Type representation."""


import typing
from typing import Tuple as TupleT

import numpy

from .utils import (
    EnvInstance,
    MyiaInputTypeError,
    SymbolicKeyInstance,
    UniverseInstance,
    register_serialize,
)

_type_cache = {}


def as_frozen(x):
    """Return an immutable representation for x."""
    if isinstance(x, dict):
        return tuple(sorted((k, as_frozen(v)) for k, v in x.items()))
    else:
        assert not isinstance(x, (list, tuple))
        return x


class TypeMeta(type):
    """Metaclass for types."""

    def __init__(cls, name, bases, dict):
        """Initialize a new Type."""
        super().__init__(name, bases, dict)
        if not hasattr(cls, '_params'):
            cls._params = None
        ann = getattr(cls, '__annotations__', {})
        cls._fields = [k for k in ann.keys()
                       if not k.startswith('_')]

    def parameterize(cls, *args):  # pragma: no cover
        """Parameterize this generic to get a concrete type.

        This is called by __getitem__.
        """
        raise TypeError('Cannot parameterize type.')
        # # The following generic implementation may be uncommented if needed.
        # fields = cls._fields
        # if len(fields) != len(args):
        #     raise TypeError('Invalid type parameterization')
        # kw = {name: arg for name, arg in zip(fields, args)}
        # return cls.make_subtype(**kw)

    def make_subtype(cls, **params):
        """Low-level parameterization function.

        The named parameters correspond to the fields declared in the Type's
        annotations.
        """
        fields = cls._fields
        if not fields:
            raise TypeError(f'{cls} cannot be parameterized.')
        elif cls._params is not None:
            raise TypeError(f'{cls} is already instantiated')
        elif list(params.keys()) != fields:
            raise TypeError('Invalid type parameterization')
        else:
            key = (cls, as_frozen(params))
            if key in _type_cache:
                return _type_cache[key]
            rval = type(cls.__qualname__, (cls,), {'_params': params})
            for k, v in params.items():
                setattr(rval, k, v)
            _type_cache[key] = rval
            return rval

    def __getitem__(cls, args):
        """Parameterize this generic type."""
        if not isinstance(args, tuple):
            args = args,
        return cls.parameterize(*args)

    def __repr__(cls):
        if cls._params is None:
            args = ''
        else:
            args = f'[{", ".join([repr(a) for a in cls._params.values()])}]'
        return f'{cls.__qualname__}{args}'


class Type(metaclass=TypeMeta):
    """Base class for all Types."""

    def __init__(self, *args, **kwargs):
        """Type cannot be initialized."""
        raise RuntimeError(f'Cannot instantiate Myia type {type(self)}.')


class Object(Type):
    """Some object."""


class Nil(Object):
    """Type of None."""


class NotImplementedType(Object):
    """Type of NotImplemented."""


class Bool(Object):
    """Boolean values."""


class String(Object):
    """String values."""


class Number(Object):
    """Numerical values.

    Cannot be parameterized directly.
    """

    bits: int
    _valid_bits: TupleT[int, ...] = ()

    @classmethod
    def parameterize(cls, bits):
        """Parameterize using a number of bits."""
        if not cls._valid_bits:
            raise RuntimeError(f"Can't parameterize {cls.__name__} directly")
        if bits not in cls._valid_bits:
            raise ValueError(f"Unsupported number of bits: {bits}")
        return cls.make_subtype(bits=bits)


class Float(Number):
    """Represents float values.

    Instantiate with `Float[nbits]`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (16, 32, 64)


class Integral(Number):
    """Represents integer values (signed or not).

    Should not be instanciated directly.
    """

    _valid_bits = (8, 16, 32, 64)


class Int(Integral):
    """Represents signed integer values.

    Instantiate with `Int[nbits]`.  Unsupported values will raise a
    ValueError.
    """


class UInt(Integral):
    """Represents unsigned integer values.

    Instantiate with `UInt[nbits]`.  Unsupported values will raise a
    ValueError.
    """


class Tuple(Object):
    """Type of a Python tuple."""


class Dict(Object):
    """Type of a Python dict."""


class NDArray(Object):
    """Type of a Numpy array."""

    @classmethod
    def to_numpy(self, x):
        """Convert ndarray x to an ndarray."""
        if not isinstance(x, numpy.ndarray):
            raise MyiaInputTypeError(f"Expected numpy.ndarray but got {x}.")
        return x

    @classmethod
    def from_numpy(self, x):
        """Convert ndarray x to an ndarray."""
        return x


class SymbolicKeyType(Object):
    """Type of a SymbolicKeyInstance."""


class EnvType(Object):
    """Represents a sensitivity map.

    This is not parameterizable, but roughly corresponds to a map from
    node identifiers to List[node.type]. It is the type of the gradient of
    a function or closure. Different closures will get and set different
    keys in this map, but the type itself is the same.
    """


class ExceptionType(Object):
    """Represents an exception."""


class UniverseType(Object):
    """Represents a collection of states."""


i8 = Int[8]
i16 = Int[16]
i32 = Int[32]
i64 = Int[64]
u8 = UInt[8]
u16 = UInt[16]
u32 = UInt[32]
u64 = UInt[64]
f16 = Float[16]
f32 = Float[32]
f64 = Float[64]


register_serialize(Bool, 'bool')
register_serialize(Nil, 'nil')
register_serialize(SymbolicKeyType, 'symbolic_key_type')
register_serialize(EnvType, 'env_type')
register_serialize(Tuple, 'tuple_type')
register_serialize(NDArray, 'ndarray_type')
register_serialize(i8, 'i8')
register_serialize(i16, 'i16')
register_serialize(i32, 'i32')
register_serialize(i64, 'i64')
register_serialize(u8, 'u8')
register_serialize(u16, 'u16')
register_serialize(u32, 'u32')
register_serialize(u64, 'u64')
register_serialize(f16, 'f16')
register_serialize(f32, 'f32')
register_serialize(f64, 'f64')
register_serialize(UniverseType, 'universe_type')

register_serialize(numpy.bool, 'numpy_bool')
register_serialize(numpy.int, 'numpy_int')
register_serialize(numpy.float, 'numpy_float')
register_serialize(numpy.int8, 'numpy_int8')
register_serialize(numpy.int16, 'numpy_int16')
register_serialize(numpy.int32, 'numpy_int32')
register_serialize(numpy.int64, 'numpy_int64')
register_serialize(numpy.uint8, 'numpy_uint8')
register_serialize(numpy.uint16, 'numpy_uint16')
register_serialize(numpy.uint32, 'numpy_uint32')
register_serialize(numpy.uint64, 'numpy_uint64')
register_serialize(numpy.float16, 'numpy_float16')
register_serialize(numpy.float32, 'numpy_float32')
register_serialize(numpy.float64, 'numpy_float64')

DTYPE_TO_MTYPE = dict(
    int8=Int[8],
    int16=Int[16],
    int32=Int[32],
    int64=Int[64],
    uint8=UInt[8],
    uint16=UInt[16],
    uint32=UInt[32],
    uint64=UInt[64],
    float16=Float[16],
    float32=Float[32],
    float64=Float[64],
    bool=Bool,
)


def np_dtype_to_type(dtype):
    """Map a numpy type string to a myia type."""
    if dtype not in DTYPE_TO_MTYPE:
        raise TypeError(f"Unsupported dtype {dtype}")
    return DTYPE_TO_MTYPE[dtype]


MTYPE_TO_DTYPE = dict((v, k) for k, v in DTYPE_TO_MTYPE.items())


def type_to_np_dtype(type):
    """Map a myia type to a numpy type string."""
    if type not in MTYPE_TO_DTYPE:
        raise TypeError(f"Can't convert to NumPy dtype {type}")
    return MTYPE_TO_DTYPE[type]


_simple_types = {
    type(None): Nil,
    type(NotImplemented): NotImplementedType,
    bool: Bool,
    str: String,
    int: Int[64],
    float: Float[64],
    numpy.int8: Int[8],
    numpy.int16: Int[16],
    numpy.int32: Int[32],
    numpy.int64: Int[64],
    numpy.uint8: UInt[8],
    numpy.uint16: UInt[16],
    numpy.uint32: UInt[32],
    numpy.uint64: UInt[64],
    numpy.float16: Float[16],
    numpy.float32: Float[32],
    numpy.float64: Float[64],
    EnvInstance: EnvType,
    SymbolicKeyInstance: SymbolicKeyType,
    UniverseInstance: UniverseType,
}


def pytype_to_myiatype(pytype):
    """Convert a Python type into a Myia type.

    Arguments:
        pytype: The Python type to convert.
    """
    return _simple_types[pytype]


#############################################
# Extra types compatible with typing module #
#############################################


T = typing.TypeVar('T')


Array = typing._GenericAlias(numpy.ndarray, T)


class External(typing.Generic[T]):
    """Represents a type external to Myia (essentially invalid)."""

    obj: T


__all__ = [
    'Bool',
    'Dict',
    'EnvType',
    'ExceptionType',
    'External',
    'Float',
    'Int',
    'NDArray',
    'Nil',
    'NotImplementedType',
    'Number',
    'Object',
    'String',
    'SymbolicKeyType',
    'Tuple',
    'Type',
    'TypeMeta',
    'UInt',
    'f16',
    'f32',
    'f64',
    'i16',
    'i32',
    'i64',
    'i8',
    'np_dtype_to_type',
    'pytype_to_myiatype',
    'type_to_np_dtype',
    'u16',
    'u32',
    'u64',
    'u8',
]
