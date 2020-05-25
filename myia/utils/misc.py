"""Miscellaneous utilities."""

import builtins
import functools
from collections import deque
from typing import Any, Dict, List, TypeVar

import numpy as np

from .serialize import serializable

builtins_d = vars(builtins)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


@serializable("RandomStateWrapper")
class RandomStateWrapper:
    """Represents a wrapper around a backend random state object."""

    __slots__ = ("state",)

    def __init__(self, backend_state):
        """Initialize wrapper with given backend random state object."""
        self.state = backend_state

    def _serialize(self):
        return {"state": self.state}

    @classmethod
    def _construct(cls):
        res = cls(None)
        data = yield res
        res.state = data["state"]


@serializable("TaggedValue")
class TaggedValue:
    """Represents a tagged value for a TaggedUnion."""

    def __init__(self, tag, value):
        """Initialize a TaggedValue."""
        self.tag = tag
        self.value = value

    def _serialize(self):
        return {"tag": self.tag, "value": self.value}

    @classmethod
    def _construct(cls):
        res = cls(None, None)
        data = yield res
        res.tag = data["tag"]
        res.value = data["value"]

    def has(self, tag):
        """Return whether this TaggedValue has the given tag."""
        return tag == self.tag

    def cast(self, tag):
        """Cast this TaggedValue to its value if the correct tag is given."""
        assert self.has(tag)
        return self.value


class Named:
    """A named object.

    This class can be used to construct objects with a name that will be used
    for the string representation.

    """

    def __init__(self, name):
        """Construct a named object.

        Arguments:
            name: The name of this object.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


UNKNOWN = Named("UNKNOWN")
MISSING = Named("MISSING")


class Registry(Dict[T1, T2]):
    """Associates primitives to implementations."""

    def __init__(self, default_field=None) -> None:
        """Initialize a Registry."""
        super().__init__()
        self.default_field = default_field

    def register(self, *prims):
        """Register a primitive."""

        def deco(fn):
            """Decorate the function."""
            for prim in prims:
                self[prim] = fn
            return fn

        return deco

    def __missing__(self, prim):
        if self.default_field and isinstance(prim, HasDefaults):
            dflt = prim.defaults()
            return dflt[self.default_field]
        raise KeyError(prim)


class HasDefaults:
    """Object that can return a defaults dictionary.

    The defaults can be given as a dictionary or as a path to a module.
    """

    def __init__(self, name, defaults, defaults_field):
        """Initialize a HasDefaults."""
        self.name = name
        self.defaults_field = defaults_field
        self.set_defaults(defaults)

    def set_defaults(self, defaults):
        """Set the defaults."""
        if isinstance(defaults, dict):
            self._defaults = defaults
        elif isinstance(defaults, str):
            self._defaults = None
            self._defaults_location = defaults
        else:
            ty = type(self).__qualname__
            raise TypeError(
                f"{ty} defaults must be a dict or the qualified name"
                " of a module."
            )

    def defaults(self):
        """Return defaults for this object."""
        if self._defaults is None:
            defaults = resolve_from_path(self._defaults_location)
            if not isinstance(defaults, dict):
                defaults = getattr(defaults, self.defaults_field)
            self._defaults = defaults
        return self._defaults


def repr_(obj: Any, **kwargs: Any):
    """Return unique string representation of object with additional info.

    The usual representation is `<module.Class object at address>`. This
    function returns `<module.Class(key=value) object at address>` instead, to
    make objects easier to identify by their attributes.

    Arguments:
        obj: object to represent
        **kwargs: The attributes and their values that will be printed as part
            of the string representation.

    """
    name = f"{obj.__module__}.{obj.__class__.__name__}"
    info = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    address = str(hex(id(obj)))
    return f"<{name}({info}) object at {address}>"


def list_str(lst: List):
    """Return string representation of a list.

    Unlike the default string representation, this calls `str` instead of
    `repr` on each element.

    """
    elements = ", ".join(str(elem) for elem in lst)
    return f"[{elements}]"


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
        return f"Event({self.name})"


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

    Arguments:
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
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f'NS({", ".join(args)})'


class Namespace:
    """Namespace in which to resolve variables."""

    def __init__(self, label, *dicts):
        """Initialize a namespace.

        Arguments:
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
        return f":{self.label}"  # pragma: no cover


class ModuleNamespace(Namespace):
    """Namespace that represents a Python module."""

    def __init__(self, name):
        """Initialize a ModuleNamespace.

        Arguments:
            name: Qualified name for the module. Must be possible
                to `__import__` it.

        """
        mod = vars(__import__(name, fromlist=["_"]))
        super().__init__(name, mod, builtins_d)


class ClosureNamespace(Namespace):
    """Namespace that represents a function's closure."""

    def __init__(self, fn):
        """Initialize a ClosureNamespace.

        Arguments:
            fn: The function.

        """
        lbl = f"{fn.__module__}..<{fn.__name__}>"
        names = fn.__code__.co_freevars
        cells = fn.__closure__
        ns = dict(zip(names, cells or ()))
        super().__init__(lbl, ns)

    def __getitem__(self, name):
        (d,) = self.dicts
        try:
            return d[name].cell_contents
        except ValueError:
            raise UnboundLocalError(name)


def is_dataclass_type(cls):
    """Returns whether cls is a dataclass."""
    return isinstance(cls, type) and hasattr(cls, "__dataclass_fields__")


def dataclass_fields(dc):
    """Returns a dataclass's fields dictionary."""
    return {name: getattr(dc, name) for name in dc.__dataclass_fields__}


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

    def trigger(self, stringify=lambda err: f"* {err.args[0]}"):
        """Raise an exception if an error occurred."""
        if self.errors:
            msg = "\n".join(stringify(e) for e in self.errors)
            exc = self.exc_class(msg)
            exc.errors = self.errors
            raise exc


def keyword_decorator(deco):
    """Wrap a decorator to optionally takes keyword arguments."""

    @functools.wraps(deco)
    def new_deco(fn=None, **kwargs):
        if fn is None:

            @functools.wraps(deco)
            def newer_deco(fn):
                return deco(fn, **kwargs)

            return newer_deco
        else:
            return deco(fn, **kwargs)

    return new_deco


@keyword_decorator
def core(fn=None, **flags):
    """Wrap a graph that defines a core Myia function.

    The following flags can be set:
        core: (default: True) Indicates that this is a core function
            (only informative at the moment).
        ignore_values: (default: False) Make the inferrer ignore argument
            values for the parameters (leads to less specialization).
    """
    flags = {
        # This is a function defined in Myia's core
        "core": True,
        "reference": True,
        **flags,
    }
    fn._myia_flags = flags
    return fn


def resolve_from_path(path):
    """Resolve a module or object from a path of the form x.y.z."""
    modname, field = path.rsplit(".", 1)
    mod = __import__(modname, fromlist=[field])
    return getattr(mod, field)


def assert_scalar(*args):
    """Assert that the arguments are all scalars."""
    # TODO: These checks should be stricter, e.g. require that all args
    # have exactly the same type, but right now there is some mixing between
    # numpy types and int/float.
    for x in args:
        if isinstance(x, np.ndarray):
            if x.shape != ():
                msg = f"Expected scalar, not array with shape {x.shape}"
                raise TypeError(msg)
        elif not isinstance(x, (int, float, np.number)):
            raise TypeError(f"Expected scalar, not {type(x)}")


class Tag:
    """A tag for dataclass fields."""

    def __init__(self, name):
        """Construct a Tag.

        Arguments:
            name: The name of this tag.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


class TagFactory:
    """A factory for Tags."""

    def __init__(self):
        """Initialize a TagFactory."""
        self.cache = {}

    def __getattr__(self, attr):
        if attr not in self.cache:
            self.cache[attr] = Tag(attr)
        return self.cache[attr]


tags = TagFactory()


class WorkSet:
    """Tool to iterate over an evolving set of objects.

    Elements can be added to a WorkSet during iteration.

    Iterating over a WorkSet will never yield the same object twice.
    """

    def __init__(self, elements):
        """Initialize a WorkSet."""
        self.done = None
        self.elements = deque(elements)

    def processed(self, element):
        """Return whether an element was processed."""
        return element in self.done

    def set_next(self, element):
        """Add an element to the beginning of the queue."""
        self.elements.appendleft(element)

    def queue(self, element):
        """Add an element to the queue."""
        self.elements.append(element)

    def queue_all(self, elements):
        """Add elements to the queue."""
        self.elements.extend(elements)

    def requeue(self, element):
        """Add an element to the queue."""
        self.done.discard(element)
        self.elements.append(element)

    def __iter__(self):
        self.done = set()
        while self.elements:
            elem = self.elements.popleft()
            if elem in self.done:
                continue
            self.done.add(elem)
            yield elem


__consolidate__ = True
__all__ = [
    "ClosureNamespace",
    "ErrorPool",
    "Event",
    "Events",
    "HasDefaults",
    "RandomStateWrapper",
    "MISSING",
    "ModuleNamespace",
    "NS",
    "Named",
    "Namespace",
    "Registry",
    "Tag",
    "TagFactory",
    "TaggedValue",
    "UNKNOWN",
    "WorkSet",
    "assert_scalar",
    "core",
    "dataclass_fields",
    "is_dataclass_type",
    "keyword_decorator",
    "list_str",
    "repr_",
    "resolve_from_path",
    "tags",
]
