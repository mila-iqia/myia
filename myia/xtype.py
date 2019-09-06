"""Type representation."""


import typing
from typing import Tuple as TupleT

import numpy

from .utils import EnvInstance, SymbolicKeyInstance

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


class Int(Number):
    """Represents signed integer values.

    Instantiate with `Int[nbits]`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (8, 16, 32, 64)


class UInt(Number):
    """Represents unsigned integer values.

    Instantiate with `UInt[nbits]`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (8, 16, 32, 64)


class Tuple(Object):
    """Type of a Python tuple."""


class Dict(Object):
    """Type of a Python dict."""


class NDArray(Object):
    """Type of a Numpy array."""


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
