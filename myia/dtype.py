"""Type representation."""


import numpy
from types import FunctionType
from typing import Tuple as TupleT, Dict as DictT, Any
from .utils import Named, is_dataclass_type, as_frozen, overload, \
    SymbolicKeyInstance, EnvInstance


_type_cache = {}


def ismyiatype(t, model=None, generic=None):
    """Check that the argument is a Myia type.

    Arguments:
        t: The object to check.
        model: If not None, check that t is the model, or a subtype
            of it.
        generic: If True, check that t is a generic. If False, check that
            t is *not* a generic. If None (default), don't check.
    """
    if not isinstance(t, TypeMeta):
        return False
    if model is not None and not issubclass(t, model):
        return False
    if generic is None:
        return True
    return t.is_generic() == generic


def get_generic(t1, t2):
    """Check that t1 and t2 are subtypes of the same generic and return it."""
    if ismyiatype(t1, generic=False) and ismyiatype(t2, generic=False) \
            and t1.generic == t2.generic:
        return t1.generic
    else:
        return None


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
        cls.generic = cls

    def is_generic(cls):
        """Return whether this type is a generic."""
        return cls.generic is cls

    def parameterize(cls, *args):
        """Parameterize this generic to get a concrete type.

        This is called by __getitem__.
        """
        fields = cls._fields
        if len(fields) != len(args):
            raise TypeError('Invalid type parameterization')
        kw = {name: arg for name, arg in zip(fields, args)}
        return cls.make_subtype(**kw)

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
            rval.generic = cls
            _type_cache[key] = rval
            return rval

    def __getitem__(cls, args):
        """Parameterize this generic type."""
        if not isinstance(args, tuple):
            args = args,
        return cls.parameterize(*args)

    def __repr__(cls):
        if hasattr(cls, '__type_repr__'):
            return cls.__type_repr__()
        name = cls.__qualname__
        if cls._params is None:
            args = ''
        else:
            args = f'[{", ".join([repr(a) for a in cls._params.values()])}]'
        return f'{name}{args}'

    def __visit__(cls, fn):
        """Visit function for unification purposes."""
        if cls.is_generic():
            fn(id(cls))
            return cls
        else:
            fn(cls.generic)
            args = {}
            if cls._params:
                for k, v in cls._params.items():
                    args[k] = fn(v)
            return cls.generic.make_subtype(**args)


class Type(metaclass=TypeMeta):
    """Base class for all Types."""

    def __init__(self, *args, **kwargs):
        """Type cannot be initialized."""
        raise RuntimeError('Cannot instantiate Myia types.')


class Object(Type):
    """Some object."""


class Bool(Object):
    """Boolean values."""


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


class List(Object):
    """Represents a set of ordered values with the same type.

    Instanciate with `List[element_type]`.
    """

    element_type: Type


class Class(Object):
    """Represents a set of methods and named fields with their own types.

    Instantiate with
    `Class[Tag, Mapping[str, Type], Mapping[str, Primitive/[Meta]Graph]]`.

    Attributes:
        tag: A unique Named identifier for this class.
        attributes: Named fields, stored in each instance.
        methods: Named methods available for this class.

    """

    tag: Named
    attributes: DictT[str, Type]
    methods: DictT[str, Any]

    @classmethod
    def __type_repr__(cls):
        args = ', '.join(f'{name}: {repr(attr)}'
                         for name, attr in cls.attributes.items())
        return f"{cls.tag}[{args}]"


class Tuple(Object):
    """Represents a set of ordered values with independent types.

    Instantiate with `Tuple[type1, type2, ... typeN]`.  A single
    sequence of types is also acceptable as the sole argument.
    """

    elements: TupleT[Type, ...]

    @classmethod
    def parameterize(cls, *elements):
        """Parameterize using a list of elements."""
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements, = elements
        return cls.make_subtype(elements=tuple(elements))

    @classmethod
    def __type_repr__(cls):
        if hasattr(cls, 'elements'):
            elems = ', '.join(map(repr, cls.elements))
        else:
            elems = ''
        return f'Tuple[{elems}]'


class Array(Object):
    """Represents an array of values.

    Instantiate with Array[subtype].
    """

    elements: Type


class Function(Object):
    """Represents a type that can be called.

    Instantiate with `Function[(type1, type2, ..., typeN), ret_type]`.
    """

    arguments: TupleT[Type, ...]
    retval: Type

    @classmethod
    def parameterize(cls, arguments, retval):
        """Parameterize using a sequence of arguments and a return type."""
        assert isinstance(arguments, (list, tuple))
        return cls.make_subtype(arguments=arguments, retval=retval)


class SymbolicKeyType(Object):
    """Type of a SymbolicKeyInstance."""


class EnvType(Object):
    """Represents a sensitivity map.

    This is not parameterizable, but roughly corresponds to a map from
    node identifiers to List[node.type]. It is the type of the gradient of
    a function or closure. Different closures will get and set different
    keys in this map, but the type itself is the same.
    """


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
    bool=Bool)


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


_simple_types = {
    bool: Bool,
    int: Int[64],
    float: Float[64],
    numpy.int8: Int[8],
    numpy.int16: Int[16],
    numpy.int32: Int[32],
    numpy.int64: Int[64],
    numpy.float16: Float[16],
    numpy.float32: Float[32],
    numpy.float64: Float[64],
}


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

    elif pytype in _simple_types:
        return _simple_types[pytype]

    elif pytype is tuple:
        if instance is None:
            return Tuple
        else:
            elems = [pytype_to_myiatype(type(x), x) for x in instance]
            return Tuple[elems]

    elif pytype is list:
        if instance is None:
            return List
        elif len(instance) == 0:
            raise TypeError('Cannot acquire the type of []')
        else:
            type0, *rest = [pytype_to_myiatype(type(x), x) for x in instance]
            if any(t != type0 for t in rest):
                raise TypeError(f'All list elements should have same type')
            return List[type0]

    elif pytype is numpy.ndarray:
        if instance is None:
            return Array
        else:
            return Array[DTYPE_MAP[instance.dtype.name]]

    elif pytype is EnvInstance:
        return EnvType

    elif pytype is SymbolicKeyInstance:
        return SymbolicKeyType

    elif is_dataclass_type(pytype):
        if pytype in dataclass_to_myiaclass:
            mcls = dataclass_to_myiaclass[pytype]
            if instance is None:
                return mcls
            tag = mcls.tag
        elif instance is None:
            tag = Named(pytype.__name__)
        else:
            tag = pytype_to_myiatype(pytype).tag

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
        rval = Class[tag, attributes, methods]
        if pytype not in dataclass_to_myiaclass:
            dataclass_to_myiaclass[pytype] = rval
            tag_to_dataclass[tag] = pytype
        return rval

    else:
        return External[pytype]


leaf_types = (Bool, Number, TypeType, Problem, External,
              EnvType, SymbolicKeyType)


@overload(bootstrap=True)
def type_cloner(self, t: leaf_types, *args):
    """Base function to clone a type recursively.

    Create a variant of this function to make type transformers.
    """
    return t


@overload  # noqa: F811
def type_cloner(self, t: List, *args):
    return List[self(t.element_type, *args)]


@overload  # noqa: F811
def type_cloner(self, t: Array, *args):
    return Array[self(t.elements, *args)]


@overload  # noqa: F811
def type_cloner(self, t: Tuple, *args):
    elt_t = [self(t2, *args) for t2 in t.elements]
    return Tuple[elt_t]


@overload  # noqa: F811
def type_cloner(self, t: Class, *args):
    return Class[
        t.tag,
        {k: self(v, *args) for k, v in t.attributes.items()},
        t.methods
    ]


@overload  # noqa: F811
def type_cloner(self, t: Function, *args):
    return Function[
        [self(t2, *args) for t2 in t.arguments],
        self(t.retval, *args)
    ]


@overload  # noqa: F811
def type_cloner(self, t: TypeMeta, *args):
    return self[t](t, *args)


@overload(bootstrap=True)
async def type_cloner_async(self, t: leaf_types, *args):
    """Base function to asynchronously clone a type recursively.

    Create a variant of this function to make type transformers.
    """
    return t


@overload  # noqa: F811
async def type_cloner_async(self, t: List, *args):
    return List[await self(t.element_type, *args)]


@overload  # noqa: F811
async def type_cloner_async(self, t: Array, *args):
    return Array[await self(t.elements, *args)]


@overload  # noqa: F811
async def type_cloner_async(self, t: Tuple, *args):
    elt_t = [await self(t2, *args) for t2 in t.elements]
    return Tuple[elt_t]


@overload  # noqa: F811
async def type_cloner_async(self, t: Class, *args):
    return Class[
        t.tag,
        {k: await self(v, *args) for k, v in t.attributes.items()},
        t.methods
    ]


@overload  # noqa: F811
async def type_cloner_async(self, t: Function, *args):
    return Function[
        [await self(t2, *args) for t2 in t.arguments],
        await self(t.retval, *args)
    ]


@overload  # noqa: F811
async def type_cloner_async(self, t: TypeMeta, *args):
    if t.is_generic():
        return t
    else:
        return await self[t](t, *args)
