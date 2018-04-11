"""Type representation."""
from typing import Tuple as TupleT, Dict as DictT, Iterable, Any
from operator import itemgetter
import collections

KeysT = Iterable[TupleT[str, 'Type']]


def make_new(types):
    """Create a __new__ method with the specified arguments."""
    args = ', '.join(f'{name}' for name in types.keys())

    ns: DictT[str, Any] = dict()
    tmpl = f"""def __new__(cls, {args}):
    return tuple.__new__(cls, ({args}{',' if len(args) != 0 else ''}))
    """
    exec(tmpl, ns)
    new = ns['__new__']
    new.__annotations__ = types
    return new


def get_types(bases):
    """Return the type of the class fields."""
    if len(bases) > 1:
        raise TypeError("Multiple inheritance not supported")
    elif len(bases) == 0:
        return {}
    elif isinstance(bases[0], TypeMeta):
        # We need to make a copy here
        return dict(bases[0]._fields)  # type: ignore
    return {}


class TypeMeta(type):
    """Metaclass for types.

    Ensures that instances are unique.
    """

    def __new__(cls, typename, bases, ns):
        """Create a new Type."""
        if ns.get('_root', False):
            bases = (tuple,) + bases
            ns['_instances'] = dict()
        if '__slots__' in ns:
            raise TypeError("Don't define __slots__")
        ns['__slots__'] = ()

        types = get_types(bases)
        types.update(filter(lambda v: not v[0].startswith('_'),
                            ns.get('__annotations__', {}).items()))
        ns['_fields'] = types

        if '__new__' not in ns:
            ns['__new__'] = make_new(types)
        for i, k in enumerate(types.keys()):
            ns[k] = property(itemgetter(i))
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

    def __new__(self, *args):
        """Disable instantiation of the base type."""
        raise RuntimeError("Can't instantiate Type")

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        name = type(self).__name__
        args = ', '.join(repr(getattr(self, p)) for p in self._fields.keys())
        return f"{name}({args})"


class Bool(Type):
    """Boolean values."""


class Number(Type):
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


class List(Type):
    """Represents a set of ordered values with the same type.

    Instanciate with `List(element_type)`.
    """

    element_type: Type


class Struct(Type):
    """Represents a set of named fields with their own types.

    Instantiate with `Struct(Mapping[str, Type])`.  A sequence of
    key-value pairs is also acceptable, but duplicate keys will will
    get lost.
    """

    elements: DictT[str, Type]

    @classmethod
    def _parse_args(cls, args, kwargs):
        if len(kwargs) != 0:
            assert len(args) == 0
            elems = kwargs
        else:
            assert len(args) == 1
            elems = dict(args[0])
        items = ((k, elems[k]) for k in sorted(elems.keys()))
        return (tuple(items),)

    def __new__(cls, elements: KeysT) -> None:
        """Convert input to a dict."""
        return tuple.__new__(cls, (dict(elements),))  # type: ignore

    def __getattr__(self, attr):
        if attr in self.elements:
            return self.elements[attr]
        raise AttributeError


class Tuple(Type):
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


class Function(Type):
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
