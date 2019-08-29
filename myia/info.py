"""Objects and routines to track debug information."""

import traceback
import types
import weakref
from contextvars import ContextVar
from typing import Any, Set


class StackVar:
    """ContextVar that represents a stack."""

    def __init__(self, name):
        """Initialize a StackVar."""
        self.var = ContextVar(name, default=(None, None))
        self.var.set((None, None))

    def push(self, x):
        """Push a new value on the stack."""
        self.var.set((x, self.var.get()))

    def pop(self):
        """Remove the top element of the stack and return it."""
        curr, prev = self.var.get()
        assert prev is not None
        self.var.set(prev)
        return curr

    def top(self):
        """Return the top element of the stack."""
        return self.var.get()[0]


_about = StackVar('_about')


def current_info():
    """Return the `DebugInfo` for the current context."""
    return _about.top()


class DebugInfo(types.SimpleNamespace):
    """Debug information for an object.

    The `DebugInherit` context manager can be used to automatically
    set certain attributes:

    >>> with DebugInherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3

    """

    def __init__(self, obj=None, **kwargs):
        """Construct a DebugInfo object."""
        top = current_info()
        if top:
            # Only need to look at the top of the stack
            self.__dict__.update(top.__dict__)

        super().__init__(**kwargs)


class DebugInherit(DebugInfo):
    """Context manager to automatically set attributes on DebugInfo.

    >>> with DebugInherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3
    """

    def __enter__(self):
        """Set this `DebugInherit` as a template in this context.

        Any `DebugInfo` created within the context of
        `with self: ...` will inherit all attributes of `self`.
        """
        _about.push(self)

    def __exit__(self, type, value, tb):
        """Exit the context of this `DebugInherit`."""
        assert current_info() is self
        _about.pop()


class NamedDebugInfo(DebugInfo):
    """Debug information for an object.

    Attributes:
        name: The name of the object.
        about: `About` object pointing to the `DebugInfo` of
            a different object, that this one is about.
        save_trace: Whether the trace of the DebugInfo's
            creation should be saved.
        trace: The trace at the moment of the DebugInfo's
            creation, or None.

    """

    _curr_id = 0

    def __init__(self, obj=None, **kwargs):
        """Construct a NamedDebugInfo object."""
        self._id: int = None
        self.name: str = None
        self.about = None
        self.errors: Set = set()
        self.save_trace: bool = False
        self.trace: Any = None
        self._obj = weakref.ref(obj) if obj else None

        super().__init__(obj, **kwargs)

        if self.save_trace:
            # We remove the last entry that corresponds to
            # this line in the code.
            self.trace = traceback.extract_stack()[:-1]

    @property
    def obj(self):
        """Return the object that this DebugInfo is about."""
        return self._obj and self._obj()

    @property
    def id(self):
        """Generate a unique, sequential ID number."""
        if self._id is None:
            self._id = self._curr_id
            NamedDebugInfo._curr_id += 1
        return self._id

    @property
    def debug_name(self):
        """Return the name, create a fresh name if needed."""
        if self.name:
            return self.name
        prefix = ''
        if self.obj is not None:
            prefix = self.obj.__class__.__name__.lower()
        self.name = f'_{prefix}{self.id}'
        return self.name

    def find(self, prop, skip=set()):
        """Find a property in self or in self.about.debug."""
        curr = self
        while curr is not None:
            rel = curr.about and curr.about.relation
            if hasattr(curr, prop) and rel not in skip:
                return getattr(curr, prop)
            if not curr.about:
                break
            curr = curr.about.debug
        return None


class About:
    """Represent a relationship to an object.

    It can be used as a context manager, in which case any `DebugInfo`
    created in the scope will have its `about` field set to the `About`
    object.

    >>> with About(x, 'purpose') as a:
    ...     info = DebugInfo()
    ...     assert info.about is a

    Attributes:
        debug: The `DebugInfo` of the object.
        relation: Some arbitrary relation with respect to the object.

    """

    def __init__(self, debug: DebugInfo, relation: str, *args) -> None:
        """Initialize an About."""
        if not isinstance(debug, DebugInfo):
            raise TypeError('debug argument to About must be a DebugInfo.') \
                # pragma: no cover
        self.debug = debug
        self.relation = relation
        self.args = args

    def __enter__(self):
        """Enter the context of this `About`."""
        _about.push(DebugInherit(about=self))

    def __exit__(self, type, value, tb):
        """Exit the context of this `About`."""
        top = _about.pop()
        assert isinstance(top, DebugInfo) and top.about is self
