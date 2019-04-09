"""Tools to intern the instances of certain classes."""

import weakref


_intern_pool = weakref.WeakValueDictionary()


class EqKey:
    """Base class for Atom/Elements."""

    def __init__(self, obj):
        """Initialize an EqKey."""
        t = type(obj)
        if t in (int, bool):
            t = float
        self.type = t


class Atom(EqKey):
    """Object with a single value to test for equality directly."""

    def __init__(self, obj, value):
        """Initialize an Atom."""
        super().__init__(obj)
        self.value = value


class Elements(EqKey):
    """Object with multiple values to process for equality recursively."""

    def __init__(self, obj, *values, values_iterable=None):
        """Initialize an Elements."""
        super().__init__(obj)
        if values_iterable is None:
            self.values = values
        else:
            assert not values
            self.values = values_iterable


def eqkey(x):
    """Return the equality key for x."""
    if isinstance(x, EqKey):
        return x
    elif isinstance(x, (list, tuple)):
        return Elements(x, *x)
    elif isinstance(x, (set, frozenset)):
        return Elements(x, values_iterable=frozenset(x))
    elif isinstance(x, dict):
        return Elements(x, *x.items())
    elif hasattr(x, '__eqkey__'):
        return x.__eqkey__()
    else:
        return Atom(x, x)


class RecursionException(Exception):
    """Raised when a data structure is found to be recursive."""


def deep_eqkey(obj):
    """Return a key for equality tests for non-recursive structures."""
    cachable = getattr(obj, '__cache_eqkey__', False)
    if cachable:
        cached = getattr(obj, '_eqkey_deepkey', None)
        if cached is not None:
            return cached

    key = eqkey(obj)
    if isinstance(key, Elements):
        dk = key.type, type(key.values)(deep_eqkey(x) for x in key.values)
    else:
        assert isinstance(key, Atom)
        dk = key.type, key.value

    if cachable:
        obj._eqkey_deepkey = dk
    return dk


def hashrec(obj):
    """Hash a (possibly self-referential) object."""
    return hash(deep_eqkey(obj))


def eqrec(obj1, obj2):
    """Compare two (possibly self-referential) objects for equality."""
    key1 = deep_eqkey(obj1)
    key2 = deep_eqkey(obj2)
    return key1 == key2


class Wrapper:
    """Wraps an object and uses eqrec/hashrec for equality."""

    def __init__(self, obj):
        """Initialize a Wrapper."""
        self._obj = weakref.ref(obj)

    def __eq__(self, other):
        return eqrec(self._obj(), other._obj())

    def __hash__(self):
        return hashrec(self._obj())


class Interned(type):
    """Represents a class where all members are interned.

    Using the __eqkey__ method to generate a key for equality purposes, each
    instance with the same eqkey is mapped to a single canonical instance.

    It is possible to create a non-interned instance with `new`.

    The `intern` method on an instance can be used to get the interned version
    of the instance.
    """

    def __init__(cls, name, bases, dct):
        """Initialize an interned class."""
        super().__init__(name, bases, dct)
        cls.intern = lambda self: Interned.intern(type(self), self)

    def new(cls, *args, **kwargs):
        """Instantiates a non-interned instance."""
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def intern(cls, inst):
        """Get the interned instance."""
        wrap = Wrapper(inst)
        existing = _intern_pool.get(wrap, None)
        if existing is None:
            _intern_pool[wrap] = inst
            return inst
        else:
            return existing

    def __call__(cls, *args, **kwargs):
        """Instantiates an interned instance."""
        inst = cls.new(*args, **kwargs)
        return inst.intern()
