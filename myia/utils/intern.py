"""Tools to intern the instances of certain classes."""

import weakref

_intern_pool = weakref.WeakValueDictionary()


pyhash = hash


class EqKey:
    """Base class for Atom/Elements."""

    def __init__(self, obj):
        """Initialize an EqKey."""
        t = type(obj)
        if t in (int, bool):
            t = float
        self.type = t
        self.obj = obj

    def canonicalize(self):
        """Canonicalize the underlying object."""


class Atom(EqKey):
    """Object with a single value to test for equality directly."""

    def __init__(self, obj, value):
        """Initialize an Atom."""
        super().__init__(obj)
        self.value = value


class ElementsBase(EqKey):
    """Object with multiple values to process for equality recursively."""


class ItemEK(ElementsBase):
    """Object indexed using getitem."""

    def __init__(self, obj, keys):
        """Initialize an ItemEK."""
        super().__init__(obj)
        self.keys = tuple(keys)
        self.values = tuple(key if isinstance(key, EqKey)
                            else obj[key]
                            for key in keys)

    def canonicalize(self):
        """Canonicalize the underlying object."""
        obj = self.obj
        for key, value in zip(self.keys, self.values):
            assert not isinstance(key, EqKey)
            obj[key] = intern(value)


class AttrEK(ElementsBase):
    """Object indexed using getattr."""

    def __init__(self, obj, keys):
        """Initialize an AttrEK."""
        super().__init__(obj)
        self.keys = tuple(keys)
        self.values = tuple(key if isinstance(key, EqKey)
                            else getattr(obj, key)
                            for key in keys)

    def canonicalize(self):
        """Canonicalize the underlying object."""
        obj = self.obj
        for key, value in zip(self.keys, self.values):
            if isinstance(key, EqKey):
                key.canonicalize()
            else:
                setattr(obj, key, intern(value))


def eqkey(x):
    """Return the equality key for x."""
    if getattr(x, '_incomplete', False):
        raise IncompleteException()
    elif isinstance(x, EqKey):
        return x
    elif isinstance(x, (list, tuple)):
        return ItemEK(x, range(len(x)))
    elif isinstance(x, dict):
        return ItemEK(x, x.keys())
    elif hasattr(x, '__eqkey__'):
        return x.__eqkey__()
    else:
        assert not isinstance(x, (set, frozenset))
        return Atom(x, x)


class RecursionException(Exception):
    """Raised when a data structure is found to be recursive."""


class IncompleteException(Exception):
    """Raised when a data structure is incomplete."""


def deep_eqkey(obj, path=frozenset()):
    """Return a key for equality tests for non-recursive structures."""
    cachable = getattr(obj, '__cache_eqkey__', False)
    if cachable:
        cached = getattr(obj, '_eqkey_deepkey', None)
        if cached is RecursionException:
            raise RecursionException()
        if cached is not None:
            return cached

    oid = id(obj)
    if oid in path:
        if cachable:
            obj._eqkey_deepkey = RecursionException
        raise RecursionException()

    key = eqkey(obj)
    if isinstance(key, ElementsBase):
        dk = key.type, type(key.values)(
            deep_eqkey(x, path | {oid}) for x in key.values
        )
    else:
        assert isinstance(key, Atom)
        dk = key.type, key.value

    if cachable:
        obj._eqkey_deepkey = dk
    return dk


def _bfs(obj):
    from collections import deque

    queue = deque()
    queue.append(obj)

    while queue:
        obj = queue.popleft()
        key = eqkey(obj)
        yield key
        if isinstance(key, ElementsBase):
            queue.extend(key.values)


def hashrec(obj, n=10):
    """Hash a (possibly self-referential) object.

    This explores the object breadth-first and uses the first n elements
    to compute the hash.

    Arguments:
        obj: The object for which to compute a hash.
        n: The maximum number of contributions to the hash.

    """
    count = 0
    h = []
    for key in _bfs(obj):
        if count == n:
            break
        count += 1
        if isinstance(key, Atom):
            h.extend((key.type, key.value))
        else:
            h.extend((key.type, len(key.values)))
    return pyhash(tuple(h))


def eqrec(obj1, obj2, cache=None):
    """Compare two (possibly self-referential) objects for equality."""
    id1 = id(obj1)
    id2 = id(obj2)

    if (id1, id2) in cache:
        return True

    cache.add((id1, id2))

    key1 = eqkey(obj1)
    key2 = eqkey(obj2)

    if type(key1) is not type(key2) or key1.type is not key2.type:
        return False

    if isinstance(key1, Atom):
        return key1.value == key2.value

    elif isinstance(key1, ElementsBase):
        v1 = key1.values
        v2 = key2.values
        if len(v1) != len(v2):
            return False
        for x1, x2 in zip(v1, v2):
            if not eqrec(x1, x2, cache):
                return False
        else:
            return True

    else:
        raise AssertionError()


def hash(obj):
    """Hash a (possibly self-referential) object."""
    try:
        return pyhash(deep_eqkey(obj))

    except RecursionException:
        return hashrec(obj)


def eq(obj1, obj2):
    """Compare two (possibly self-referential) objects for equality."""
    if obj1 is obj2:
        return True

    try:
        key1 = deep_eqkey(obj1)
        key2 = deep_eqkey(obj2)
        return key1 == key2

    except RecursionException:
        return eqrec(obj1, obj2, set())


class Wrapper:
    """Wraps an object and uses eq/hash for equality."""

    def __init__(self, obj):
        """Initialize a Wrapper."""
        self._obj = weakref.ref(obj)

    def __eq__(self, other):
        return eq(self._obj(), other._obj())

    def __hash__(self):
        return hash(self._obj())


class InternedMC(type):
    """Metaclass for a class where all members are interned."""

    def new(cls, *args, **kwargs):
        """Instantiates a non-interned instance."""
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def intern(cls, inst):
        """Get the interned instance."""
        return intern(inst)

    def __call__(cls, *args, **kwargs):
        """Instantiates an interned instance."""
        inst = cls.new(*args, **kwargs)
        return inst.intern()


class Interned(metaclass=InternedMC):
    """Instances of this class are interned.

    Using the __eqkey__ method to generate a key for equality purposes, each
    instance with the same eqkey is mapped to a single canonical instance.
    """

    def intern(self):
        """Get the interned version of the instance."""
        return InternedMC.intern(type(self), self)

    def __eqkey__(self):
        """Generate a key for equality/hashing purposes."""
        raise NotImplementedError('Implement in subclass')


class PossiblyRecursive:
    """Base class for data that might be recursive."""

    @classmethod
    def empty(cls):
        """Create an empty, incomplete instance."""
        inst = object.__new__(cls)
        inst._incomplete = True
        return inst

    def __init__(self):
        """Initialization sets the object to complete."""
        self._incomplete = False


def intern(inst):
    """Get the interned instance."""
    try:
        wrap = Wrapper(inst)
    except TypeError:
        wrap = None
        existing = None
    else:
        existing = _intern_pool.get(wrap, None)

    if existing is None:
        if wrap is not None:
            _intern_pool[wrap] = inst
        eqk = eqkey(inst)
        eqk.canonicalize()
        return inst
    else:
        return existing


__consolidate__ = True
__all__ = [
    'Atom',
    'AttrEK',
    'EqKey',
    'Interned',
    'ItemEK',
    'PossiblyRecursive',
    'eqrec',
    'hashrec',
    'intern',
]
