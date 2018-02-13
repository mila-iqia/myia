"""General utilities and design patterns."""

from typing import Any, List, Dict, TypeVar
from collections import defaultdict


T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Named:
    """A named object.

    This class can be used to construct objects with a name that will be used
    for the string representation.

    """

    def __init__(self, name):
        """Construct a named object.

        Args:
            name: The name of this object.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


class Registry(Dict[T1, T2]):
    """Associates primitives to implementations."""

    def __init__(self) -> None:
        """Initialize a Registry."""
        super().__init__()

    def register(self, prim):
        """Register a primitive."""
        def deco(fn):
            """Decorate the function."""
            self[prim] = fn
            return fn
        return deco


def memoize_method(fn):
    """Memoize the result of a method.

    The function's first argument must be `self` (in other words, it must be a
    method).

    The cache is stored per-instance, in ``self._cache[fn]``.
    """
    def deco(self, *args):
        if not hasattr(self, '_cache'):
            self._cache = defaultdict(dict)
        cache = self._cache[fn]
        if args not in self._cache:
            cache[args] = fn(self, *args)
        return cache[args]

    return deco


def repr_(obj: Any, **kwargs: Any):
    """Return unique string representation of object with additional info.

    The usual representation is `<module.Class object at address>`. This
    function returns `<module.Class(key=value) object at address>` instead, to
    make objects easier to identify by their attributes.

    Args:
        **kwargs: The attributes and their values that will be printed as part
            of the string representation.

    """
    name = f'{obj.__module__}.{obj.__class__.__name__}'
    info = ', '.join(f'{key}={value}' for key, value in kwargs.items())
    address = str(hex(id(obj)))
    return f'<{name}({info}) object at {address}>'


def list_str(lst: List):
    """Return string representation of a list.

    Unlike the default string representation, this calls `str` instead of
    `repr` on each element.

    """
    elements = ', '.join(str(elem) for elem in lst)
    return f'[{elements}]'


def _sequence_map(smap, *seqs):
    """Structural map on a sequence (list, tuple, etc.)."""
    s0 = seqs[0]
    t = type(s0)
    # Each sequence must have the same type and the same length.
    smap.require_same([type, len], seqs)
    return t(smap(*[s[i] for s in seqs]) for i in range(len(s0)))


default_smap_dispatch = {
    tuple: _sequence_map,
    list: _sequence_map
}


class StructuralMap:
    """Map a function recursively over all scalars in a structure.

    Attributes:
        fn: The function to map.
        dispatch: A type->handler dictionary to determine how to
            map structurally over that type. A handler is given the
            StructuralMap instance as its first argument, and then
            the arguments to the function.

    """

    def __init__(self, fn, dispatch=default_smap_dispatch):
        """Initialize a StructuralMap."""
        self.fn = fn
        self.dispatch = dispatch

    def require_same(self, fns, objs):
        """Check that all objects have the same properties.

        Arguments:
            fns: A collection of functions. Each function must return the same
                result when applied to each object. For example, the functions
                may be `[type, len]`.
            objs: Objects that must be invariant with respect to the given
                functions.
        """
        o, *rest = objs
        for fn in fns:
            for obj in rest:
                if fn(o) != fn(obj):
                    raise TypeError("Arguments to 'smap' do not"
                                    f" have the same properties:"
                                    f" `{o}` and `{obj}` are not conformant.")

    def __call__(self, *data):
        """Apply the StructuralMap on data.

        Each item in data corresponds to one argument of self.fns.
        """
        d0 = data[0]
        t = type(d0)
        if t in self.dispatch:
            return self.dispatch[t](self, *data)
        # elif hasattr(d0, '__smap__'):
        #     return d0.__smap__(self, *data[1:])
        else:
            return self.fn(*data)


def smap(fn, *args):
    """Map a function recursively over all scalars in a structure."""
    return StructuralMap(fn)(*args)
