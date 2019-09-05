"""Miscellaneous utilities."""

import builtins
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar

builtins_d = vars(builtins)


T1 = TypeVar('T1')
T2 = TypeVar('T2')


class ADT:
    """Base class for an algebraic data type."""


@dataclass  # pragma: no cover
class Slice:
    """Myia version of a slice."""

    start: object
    stop: object
    step: object


@dataclass  # pragma: no cover
class Cons(ADT):
    """Cons cell for lists.

    Attributes:
        head: The first element of the list.
        tail: The rest of the list.

    """

    head: object
    tail: 'Cons'

    def _to_list(self):
        curr = self
        rval = []
        while not isinstance(curr, Empty):
            rval.append(curr.head)
            curr = curr.tail
        return rval

    def __len__(self):
        return 1 + len(self.tail)

    def __getitem__(self, idx):
        if idx == 0:
            return self.head
        else:
            return self.tail[idx - 1]

    def __iter__(self):
        return iter(self._to_list())

    def __myia_iter__(self):
        return self

    def __myia_hasnext__(self):
        return True

    def __myia_next__(self):
        return self.head, self.tail


@dataclass  # pragma: no cover
class Empty(ADT):
    """Empty list."""

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise Exception('Index out of bounds')

    def __myia_iter__(self):
        return self

    def __myia_next__(self):
        raise Exception('Out of bounds')

    def __myia_hasnext__(self):
        return False


def list_to_cons(elems):
    """Convert a list to a linked list using Cons."""
    rval = Empty()
    for elem in reversed(elems):
        rval = Cons(elem, rval)
    return rval


class TaggedValue:
    """Represents a tagged value for a TaggedUnion."""

    def __init__(self, tag, value):
        """Initialize a TaggedValue."""
        self.tag = tag
        self.value = value

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

        Args:
            name: The name of this object.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


UNKNOWN = Named('UNKNOWN')
MISSING = Named('MISSING')


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
        except ValueError:
            raise UnboundLocalError(name)


def is_dataclass_type(cls):
    """Returns whether cls is a dataclass."""
    return isinstance(cls, type) and hasattr(cls, '__dataclass_fields__')


def dataclass_fields(dc):
    """Returns a dataclass's fields dictionary."""
    return {name: getattr(dc, name)
            for name in dc.__dataclass_fields__}


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
        'core': True,
        'reference': True,
        **flags,
    }
    fn._myia_flags = flags
    return fn
