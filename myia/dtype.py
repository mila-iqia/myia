"""Type representation."""

import collections
import numpy
from typing import Any, Dict as DictT, Iterable, Tuple as TupleT
from types import FunctionType
from .utils import Named, is_dataclass

KeysT = Iterable[TupleT[str, 'Type']]


def get_types(bases):
    """Return the type of the class fields."""
    if len(bases) > 1:
        raise TypeError("Multiple inheritance not supported")
    elif len(bases) == 0:
        return {}
    elif isinstance(bases[0], TypeMeta):
        # We need to make a copy here
        return dict(bases[0]._fields)
    return {}  # pragma: no cover


class TypeMeta(type):
    """Metaclass for types.

    Ensures that instances are unique.
    """

    def __new__(cls, typename, bases, ns):
        """Create a new Type."""
        if ns.get('_root', False):
            ns['_instances'] = dict()
        if '__slots__' in ns:
            raise TypeError("Don't define __slots__")

        types = get_types(bases)
        types.update(filter(lambda v: not v[0].startswith('_'),
                            ns.get('__annotations__', {}).items()))
        ns['_fields'] = types
        ns['__slots__'] = list(types.keys())

        if '__init__' not in ns:
            def init(self, *args):
                assert len(args) == len(types)
                for k, arg in zip(types.keys(), args):
                    setattr(self, k, arg)

            ns['__init__'] = init

        return type.__new__(cls, typename, bases, ns)

    def __call__(self, *args, **kwargs):
        """Overloaded to change the semantics of class instantiation."""
        if hasattr(self, '_parse_args'):
            args = self._parse_args(args, kwargs)
        else:
            assert len(kwargs) == 0
        key = (self, args)
        if key not in self._instances:
            obj = super().__call__(*args)
            self._instances[key] = obj
        return self._instances[key]

    def __str__(self):  # pragma: no cover
        return self.__name__


class Type(metaclass=TypeMeta):
    """Base class for all Types.

    This class brings a number of unusual behaviour for its subclasses
    compared to normal Python classes.

      * Arguments passed to the constructor will be passed to the
       `_parse_args(cls, args, kwargs)` class method if it exists.
       That method must return a tuple of values, one for each
       declared class attribute.

      * Instances are unique based on the parsed version of the
        constructor arguments. This means that `is` and `==` mean the
        same thing.

      * If the `__new__` method is not defined, one will be generated
        which takes one parameter for each declared class attribute
        that does not start with an underscore (`_`) and stores them
        in a read-only container that can be referred to by attribute
        access.

    Notes
    -----
        In the current implementation, Type is a subtype of tuple via
        metaclass magic, but this may change in the future.

    """

    _root = True
    _fields: DictT[str, Any]

    def __init__(self, *args):
        """Disable instantiation of the base type."""
        raise RuntimeError("Can't instantiate Type")

    def __repr__(self) -> str:
        name = type(self).__name__
        args = ', '.join(repr(getattr(self, p)) for p in self._fields.keys())
        return f"{name}({args})"


class Object(Type):
    """Some object."""

    @classmethod
    def _parse_args(cls, args, kwargs):
        if cls is Object:
            raise RuntimeError("Can't instantiate Object directly")
        return args


class Bool(Object):
    """Boolean values."""


class Number(Object):
    """Numerical values."""

    bits: int
    _valid_bits: TupleT[int, ...] = ()

    @classmethod
    def _parse_args(cls, args, kwargs):
        if cls is Number:
            raise RuntimeError("Can't instantiate Number directly")
        assert len(kwargs) == 0
        assert len(args) == 1
        if args[0] not in cls._valid_bits:
            raise ValueError(f"Unsupported number of bits: {args[0]}")
        return args


class Float(Number):
    """Represents float values.

    Instantiate with `Float(nbits)`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (16, 32, 64)


class Int(Number):
    """Represents signed integer values.

    Instantiate with `Int(nbits)`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (8, 16, 32, 64)


class UInt(Number):
    """Represents unsigned integer values.

    Instantiate with `UInt(nbits)`.  Unsupported values will raise a
    ValueError.
    """

    _valid_bits = (8, 16, 32, 64)


class List(Object):
    """Represents a set of ordered values with the same type.

    Instanciate with `List(element_type)`.
    """

    element_type: Type


class Class(Object):
    """Represents a set of methods and named fields with their own types.

    Instantiate with
    `Class(Tag, Mapping[str, Type], Mapping[str, Primitive/[Meta]Graph])`.

    Attributes:
        tag: A unique Named identifier for this class.
        attributes: Named fields, stored in each instance.
        methods: Named methods available for this class.

    """

    tag: Named
    attributes: DictT[str, Type]
    methods: DictT[str, Any]

    @classmethod
    def _parse_args(cls, args, kwargs):
        assert not kwargs
        tag, attributes, methods = args
        return tag, tuple(attributes.items()), tuple(methods.items())

    def __init__(self, tag, attributes, methods):
        """Initialize a Class."""
        self.tag = tag
        self.attributes = dict(attributes)
        self.methods = dict(methods)

    def __repr__(self) -> str:
        args = ', '.join(f'{name}: {repr(attr)}'
                         for name, attr in self.attributes.items())
        return f"{self.tag}({args})"


class Tuple(Object):
    """Represents a set of ordered values with independent types.

    Instantiate with `Tuple(type1, type2, ... typeN)`.  A single
    sequence of types is also acceptable as the sole argument.
    """

    elements: TupleT[Type, ...]

    @classmethod
    def _parse_args(cls, args, kwargs):
        assert len(kwargs) == 0
        if (len(args) == 1 and isinstance(args[0], collections.Iterable) and
                not isinstance(args[0], Type)):
            return (tuple(args[0]),)
        else:
            return (args,)


class Array(Object):
    """Represents an array of values.

    Instantiate with Array(subtype).
    """

    elements: Type


class Function(Object):
    """Represents a type that can be called.

    Instantiate with `Function((type1, type2, ..., typeN), ret_type)`.
    """

    arguments: TupleT[Type, ...]
    retval: Type

    @classmethod
    def _parse_args(cls, args, kwargs):
        assert len(args) == 2
        assert len(kwargs) == 0
        assert not isinstance(args[0], Type)
        return (tuple(args[0]), args[1])


class TypeType(Type):
    """The type of a Type."""


class Problem(Type):
    """This represents some kind of problematic type.

    For example, when the specializer tries to specialize a graph that is not
    called anywhere, it won't have the information it needs to do that, so it
    may produce the type Problem(DEAD). A Problem type may not end up being
    a real problem: dead code won't be called anyway, so it doesn't matter if
    we can't type it. Others may be real problems, e.g. Problem(POLY) which
    happens when there are multiple ways to type a graph in a given context.
    """

    kind: Any


class External(Type):
    """Represents a type external to Myia (essentially invalid)."""

    t: Any


DTYPE_MAP = dict(
    int8=Int(8),
    int16=Int(16),
    int32=Int(32),
    int64=Int(64),
    uint8=UInt(8),
    uint16=UInt(16),
    uint32=UInt(32),
    uint64=UInt(64),
    float16=Float(16),
    float32=Float(32),
    float64=Float(64),
    bool=Bool())


def np_dtype_to_type(dtype):
    """Map a numpy type string to a myia type."""
    if dtype not in DTYPE_MAP:
        raise TypeError(f"Unsupported dtype {dtype}")
    return DTYPE_MAP[dtype]


TYPE_MAP = dict((v, k) for k, v in DTYPE_MAP.items())


def type_to_np_dtype(type):
    """Map a myia type to a numpy type string."""
    if type not in TYPE_MAP:
        raise TypeError(f"Con't convert to NumPy dtype {type}")
    return TYPE_MAP[type]


dataclass_to_myiaclass = {}
tag_to_dataclass = {}


def pytype_to_myiatype(pytype, instance=None):
    """Convert a Python type into a Myia type.

    Arguments:
        pytype: The Python type to convert.
        instance: Optionally, an instance of the Python type to use
            in order to get a more precise type.
    """
    if isinstance(pytype, Type) \
            or isinstance(pytype, type) and issubclass(pytype, Type):
        return pytype

    elif pytype is bool:
        return Bool()

    elif pytype is int:
        return Int(64)

    elif pytype is float:
        return Float(64)

    elif pytype is tuple:
        if instance is None:
            return Tuple
        else:
            elems = [pytype_to_myiatype(type(x), x) for x in instance]
            return Tuple(elems)

    elif pytype is list:
        if instance is None:
            return List
        else:
            type0, *rest = [pytype_to_myiatype(type(x), x) for x in instance]
            if any(t != type0 for t in rest):
                raise TypeError(f'All list elements should have same type')
            return List(type0)

    elif pytype is numpy.ndarray:
        if instance is None:
            return Array
        else:
            return Array(DTYPE_MAP[instance.dtype.name])

    elif is_dataclass(pytype):
        if pytype in dataclass_to_myiaclass:
            mcls = dataclass_to_myiaclass[pytype]
            if instance is None:
                return mcls
            tag = mcls.tag
        else:
            tag = Named(pytype.__name__)

        fields = pytype.__dataclass_fields__
        if instance is None:
            attributes = {name: pytype_to_myiatype(field.type)
                          for name, field in fields.items()}
        else:
            attributes = {}
            for name, field in fields.items():
                x = getattr(instance, field.name)
                t = pytype_to_myiatype(type(x), x)
                attributes[field.name] = t
        methods = {name: getattr(pytype, name)
                   for name in dir(pytype)
                   if isinstance(getattr(pytype, name), (FunctionType,))}
        rval = Class(tag, attributes, methods)
        dataclass_to_myiaclass[pytype] = rval
        tag_to_dataclass[tag] = pytype
        return rval

    else:
        return External(pytype)
