"""Objects and routines to track debug information."""

from typing import Any
import types
import threading
import traceback


# We use per-thread storage for the about stack.
_about = threading.local()
_about.stack = [None]


def current_info():
    """Return the `DebugInfo` for the current context."""
    return _about.stack[-1]


class DebugInfo(types.SimpleNamespace):
    """Debug information for an object.

    If used with the `with` statement, any `DebugInfo` created
    while inside the `with` body will inherit all attributes:

        with DebugInfo(a=1, b=2):
            info = DebugInfo(c=3)
            assert info.a == 1
            assert info.b == 2
            assert info.c == 3

    Attributes:
        about: DebugInfo of a different object, that this
            object is derived from in some way
        relation: How the object relates to `about`
        save_trace: Whether the trace of the DebugInfo's
            creation should be saved.
        trace: The trace at the moment of the DebugInfo's
            creation, or None.

    """

    def __init__(self, **kwargs):
        """Construct a DebugInfo object."""
        self.about = None
        self.relation = None
        self.save_trace: bool = False
        self.trace: Any = None
        top = current_info()
        if top:
            # Only need to look at the top of the stack
            self.__dict__.update(top.__dict__)

        super().__init__(**kwargs)

        if self.save_trace:
            # We remove the last entry that corresponds to
            # this line in the code.
            self.trace = traceback.extract_stack()[:-1]

    def __enter__(self):
        """Set this `DebugInfo` as a template in this context.

        Any `DebugInfo` created within the context of
        `with self: ...` will inherit all attributes of `self`.
        """
        _about.stack.append(self)

    def __exit__(self, type, value, tb):
        """Exit the context of this `DebugInfo`."""
        assert _about.stack[-1] is self
        _about.stack.pop()


class NamedDebugInfo(DebugInfo):
    """Debug information for an object.

    Attributes:
        type: The type name of the object.
        name: The name of the object.

    """

    _curr_id = 0

    def __init__(self, **kwargs):
        """Construct an NamedDebugInfo object."""
        self._id: int = None
        self.name: str = None
        self.type: str = None
        super().__init__(**kwargs)

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
        prefix = self.type or ''
        self.name = f'_{prefix}{self.id}'
        return self.name


def about(obj, relation=None):
    """Set context as being about the given object.

    `about(obj, rel)` is equivalent to
    `DebugInfo(about=obj, relation=rel)` and can be used as
    a context manager:

        with about(x, 'purpose'):
            info = DebugInfo(a=3)
            assert info.about is x
            assert info.relation == 'purpose'
            assert info.a == 3
    """
    return DebugInfo(about=obj, relation=relation)
