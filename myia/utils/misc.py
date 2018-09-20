"""Miscellaneous utilities."""

import builtins
import inspect
import sys
from typing import Any, Dict, List, TypeVar
from colorama import AnsiToWin32


builtins_d = vars(builtins)


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


UNKNOWN = Named('UNKNOWN')


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


class TypeMap(dict):
    """Map types to handlers or values.

    Mapping a type to a value also (lazily) maps all of its subclasses to the
    same value, unless they have a mapping of their own.

    TypeMap should ideally not be updated after it is used, because updates may
    make some cached associations invalid.

    Attributes:
        discover: A function that takes a class and generates/returns a handler
            for it, if none is found.

    """

    def __init__(self, *args, discover=None):
        """Initialize a TypeMap."""
        super().__init__(*args)
        self.discover = discover

    def register(self, *obj_ts):
        """Decorator to register a handler to the given types."""
        def deco(handler):
            for obj_t in obj_ts:
                self[obj_t] = handler
            return handler
        return deco

    def __missing__(self, obj_t):
        """Get the handler for the given type."""
        handler = None
        to_set = []

        for cls in type.mro(obj_t):
            handler = super().get(cls, None)
            if handler is None and self.discover:
                handler = self.discover(cls)
            if handler is not None:
                for cls2 in to_set:
                    self[cls2] = handler
                break
            to_set.append(cls)

        if handler is not None:
            return handler
        else:
            raise KeyError(obj_t)


class Overload:
    """Overloaded function.

    A function can be added with the `register` method. One of its parameters
    should be annotated with a type, but only one, and every registered
    function should annotate the same parameter.

    Arguments:
        fallback_method: If no function is registered for a type, we will try
            to call this method on it.
        bind_to: Binds the first argument to the given object.
    """

    def __init__(self, fallback_method=None, bind_to=None, _parent=None):
        """Initialize an Overload."""
        self.__self__ = bind_to
        if _parent:
            self.map = _parent.map
            self.which = _parent.which
            return
        if fallback_method:
            self.map = TypeMap(
                discover=lambda cls: getattr(cls, fallback_method, None)
            )
        else:
            self.map = TypeMap()
        self.which = None

    def register(self, fn):
        """Register a function."""
        ann = fn.__annotations__
        if len(ann) != 1:
            raise Exception('Only one parameter may be annotated.')
        argnames = inspect.getfullargspec(fn).args
        for i, name in enumerate(argnames):
            t = ann.get(name, None)
            if t is not None:
                if isinstance(t, str):
                    t = eval(t)
                if self.which is None:
                    self.which = i
                elif self.which != i:
                    raise Exception(
                        'Annotation must always be on same parameter'
                    )
                break

        ts = t if isinstance(t, tuple) else (t,)

        for t in ts:
            self.map[t] = fn

        return self

    def __get__(self, obj, cls):
        return Overload(bind_to=obj, _parent=self)

    def __getitem__(self, t):
        if self.__self__:
            return self.map[t].__get__(self.__self__)
        else:
            return self.map[t]

    def __call__(self, *args):
        """Call the overloaded function."""
        if self.__self__:
            main = args[self.which - 1]
            return self.map[type(main)](self.__self__, *args)
        else:
            main = args[self.which]
            return self.map[type(main)](*args)


def overload(fn=None, *, fallback_method=None):
    """Overload a function.

    Overloading is based on the function name.

    The decorated function should have one parameter annotated with a type.
    Any parameter can be annotated, but only one, and every overloading of a
    function should annotate the same parameter.

    The decorator optionally takes keyword arguments, *only* on the first
    use.

    Arguments:
        fallback_method: If no function is registered for a type, we will
            try to call this method on it.
    """
    if fn is None:
        def deco(fn):
            return overload(fn, fallback_method=fallback_method)
        return deco
    mod = __import__(fn.__module__, fromlist='_')
    dispatch = getattr(mod, fn.__name__, None)
    if dispatch is None:
        dispatch = Overload(fallback_method=fallback_method)
    elif fallback_method is not None:
        raise ValueError(
            'Only the first use of @overload can take keyword arguments.'
        )
    if not isinstance(dispatch, Overload):
        raise TypeError('@overload requires Overload instance')
    return dispatch.register(fn)


def _object_map(smap, *args):
    return smap.fn(*args)


def _sequence_map(smap, *seqs):
    """Structural map on a sequence (list, tuple, etc.)."""
    s0 = seqs[0]
    t = type(s0)
    # Each sequence must have the same type and the same length.
    smap.require_same([type, len], seqs)
    return t(smap(*[s[i] for s in seqs]) for i in range(len(s0)))


default_smap_dispatch = TypeMap({
    tuple: _sequence_map,
    list: _sequence_map,
    object: _object_map,
    type: _object_map,
})


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
        return self.dispatch[t](self, *data)


def smap(fn, *args):
    """Map a function recursively over all scalars in a structure."""
    return StructuralMap(fn)(*args)


class Event:
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
        self._handlers = []
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
        self._handlers.append(handler)
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

    def remove(self, handler):
        """Remove this handler."""
        self._handlers.remove(handler)

    def __iter__(self):
        """Iterate over all handlers in the order they were added."""
        return iter(self._handlers)

    def __call__(self, *args, **kwargs):
        """Fire an event with the given arguments and keyword arguments."""
        for f in self:
            f(self, *args, **kwargs)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Event({self.name})'


class Events:
    """A group of named events.

    >>> events = Events(tickle=None, yawn=None)
    >>> events.tickle.register(lambda event, n: print("hi" * n))
    <function>
    >>> events.yawn.register(lambda event, n: print("a" * n))
    <function>
    >>> events.tickle(5)
    hihihihihi
    >>> events.yawn(20)
    aaaaaaaaaaaaaaaaaaaa

    Args:
        owner: The object that owns all the events. It will be passed
            as the first argument on each event.
        events: Keyword arguments mapping an event name to a history,
            or None if there is no history.
    """

    def __init__(self, owner=None, **events):
        """Initialize Events."""
        self.owner = owner
        for event_name, history in events.items():
            ev = Event(event_name, owner=owner, history=history)
            setattr(self, event_name, ev)


class NS:
    """Simple namespace, acts a bit like a dict with dot access on keys.

    This is different from Namespace below.

    This namespace preserves key order in the representation, unlike
    types.SimpleNamespace.
    """

    def __init__(self, **kwargs):
        """Initialize NS."""
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __repr__(self):
        args = [f'{k}={v}' for k, v in self.__dict__.items()]
        return f'NS({", ".join(args)})'


class Namespace:
    """Namespace in which to resolve variables."""

    def __init__(self, label, *dicts):
        """Initialize a namespace.

        Args:
            label: The namespace's name.
            dicts: A list of dictionaries containing the namespace's data,
                which are consulted sequentially.
        """
        self.label = label
        self.dicts = dicts

    def __contains__(self, name):
        for d in self.dicts:
            if name in d:
                return True
        else:
            return False

    def __getitem__(self, name):
        for d in self.dicts:
            if name in d:
                return d[name]
        else:
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
        super().__init__(name, mod, builtins_d)


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
        d, = self.dicts
        try:
            return d[name].cell_contents
        except ValueError as e:
            raise UnboundLocalError(name)


stderr = AnsiToWin32(sys.stderr).stream


def eprint(*things):
    """Print to stderr."""
    print(*things, file=stderr)


def is_dataclass_type(cls):
    """Returns whether cls is a dataclass."""
    return isinstance(cls, type) and hasattr(cls, '__dataclass_fields__')


def as_frozen(x):
    """Return an immutable representation for x."""
    if isinstance(x, dict):
        return tuple(sorted((k, as_frozen(v)) for k, v in x.items()))
    elif isinstance(x, (list, tuple)):
        return tuple(as_frozen(y) for y in x)
    else:
        return x


class ErrorPool:
    """Accumulates a list of errors.

    Arguments:
        exc_class: The exception to raise if any error occurred.
    """

    def __init__(self, *, exc_class=Exception):
        """Initialize the ErrorPool."""
        self.errors = []
        self.exc_class = exc_class

    def add(self, error):
        """Add an exception to the pool."""
        assert isinstance(error, Exception)
        self.errors.append(error)

    def trigger(self, stringify=lambda err: f'* {err.args[0]}'):
        """Raise an exception if an error occurred."""
        if self.errors:
            msg = "\n".join(stringify(e) for e in self.errors)
            raise self.exc_class(msg)
