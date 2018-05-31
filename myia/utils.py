"""General utilities and design patterns."""

from collections import defaultdict
from typing import Any, Dict, List, TypeVar

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
        if args not in cache:
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


class Event(list):
    """Simple Event class.

    >>> e = Event('tickle')
    >>> e.register(lambda event, n: print("hi" * n))
    <function>
    >>> e(5)
    hihihihihi

    Attributes:
        name: The name of the event.
        owner: The object defining this event (defaults to None).
        history: A function that returns a list of previous events, or
            None if no history exists.

    """

    def __init__(self, name, owner=None, history=None):
        """Initialize an Event."""
        self.name = name
        self.owner = owner
        self.history = history

    def register(self, handler, run_history=False):
        """Register a handler for this event.

        This returns the handler so that register can be used as a decorator.

        Arguments:
            handler: A function. The handler's first parameter will always
                be the event.
            run_history: Whether to call the handler for all previous events
                or not.
        """
        self.append(handler)
        if run_history and self.history:
            for entry in self.history():
                if isinstance(entry, tuple):
                    handler(self, *entry)
                else:
                    handler(self, entry)
        return handler

    def register_with_history(self, handler):
        """Register a handler for this event and run it on the history."""
        return self.register(handler, True)

    def __call__(self, *args, **kwargs):
        """Fire an event with the given arguments and keyword arguments."""
        for f in self:
            f(self, *args, **kwargs)

    def __str__(self):
        return f'Event({self.name})'

    def __repr__(self):
        return str(self)


class Events:
    """A group of named events.

    >>> events = Events('tickle', 'yawn')
    >>> events.tickle.register(lambda event, n: print("hi" * n))
    <function>
    >>> events.yawn.register(lambda event, n: print("a" * n))
    <function>
    >>> events.tickle(5)
    hihihihihi
    >>> events.yawn(20)
    aaaaaaaaaaaaaaaaaaaa
    """

    def __init__(self, owner=None, **events):
        """Initialize Events."""
        self.owner = owner
        for event_name, history in events.items():
            ev = Event(event_name, owner=owner, history=history)
            setattr(self, event_name, ev)


class Namespace:
    """Namespace in which to resolve variables."""

    def __init__(self, label, ns):
        """Initialize a namespace.

        Args:
            label: The namespace's name.
            ns: A dictionary containing the namespace's data.
        """
        self.label = label
        self.ns = ns

    def __contains__(self, name):
        return name in self.ns

    def __getitem__(self, name):
        try:
            return self.ns[name]
        except KeyError as e:
            raise NameError(name)

    def __repr__(self):
        return f':{self.label}'  # pragma: no cover


class ModuleNamespace(Namespace):
    """Namespace that represents a Python module."""

    def __init__(self, name):
        """Initialize a ModuleNamespace.

        Args:
            name: Qualified name for the module. Must be possible
                to `__import__` it.
        """
        mod = vars(__import__(name, fromlist=['_']))
        super().__init__(name, mod)


class ClosureNamespace(Namespace):
    """Namespace that represents a function's closure."""

    def __init__(self, fn):
        """Initialize a ClosureNamespace.

        Args:
            fn: The function.

        """
        lbl = f'{fn.__module__}..<{fn.__name__}>'
        names = fn.__code__.co_freevars
        cells = fn.__closure__
        ns = dict(zip(names, cells or ()))
        super().__init__(lbl, ns)

    def __getitem__(self, name):
        try:
            return self.ns[name].cell_contents
        except ValueError as e:
            raise UnboundLocalError(name)
