"""Objects and routines to track debug information."""

import traceback
import weakref
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count


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


_stack = StackVar("_stack")
_debug = ContextVar("debug", default=False)


@contextmanager
def enable_debug():
    """Enable debugging for a context."""
    tok = _debug.set(True)
    try:
        yield
    finally:
        _debug.reset(tok)


def get_debug():
    """Return whether debug is enabled or not."""
    return _debug.get()


def current_info():
    """Return the `DebugInfo` for the current context."""
    return _stack.top()


class DebugInfo:
    """Debug information for an object.

    The `debug_inherit` context manager can be used to automatically
    set certain attributes:

    >>> with debug_inherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3

    """

    def __init__(self, obj=None, __no_current=False, **kwargs):
        self.name = None
        self.about = None
        self.relation = None
        self.save_trace = False
        self.trace = None
        self._obj = None

        if not __no_current:
            top = current_info()
            if top:
                # Only need to look at the top of the stack
                self.__dict__.update(top.__dict__)
            self.__dict__.update(kwargs)

        if obj is not None:
            self._obj = weakref.ref(obj)

        if self.save_trace:
            # We remove the last entry that corresponds to
            # this line in the code.
            self.trace = traceback.extract_stack()[:-1]

    @property
    def obj(self):
        """Return the object that this DebugInfo is about."""
        return self._obj and self._obj()

    def find(self, prop, skip=set()):
        """Find a property in self or in self.about."""
        for debug, rel in self.get_chain():
            if hasattr(debug, prop) and rel not in skip:
                return getattr(debug, prop)
        else:
            return None

    def get_chain(self):
        """Return the chain of (info, relation) following the "about" field."""
        curr = self
        rval = []
        while curr is not None:
            rval.append((curr, curr.relation))
            curr = curr.about
        return rval


def make_debug(obj=None, **kwargs):
    """Returns either None or a DebugInfo, if in debug mode or not."""
    if not _debug.get():
        return None
    else:
        return DebugInfo(obj, **kwargs)


def clone_debug(dbg, objmap):
    """Clone the debug information chain for an object."""
    if dbg is None:
        return None
    old = dbg._obj()
    obj = objmap.get(old, old)
    res = DebugInfo(obj, __no_current=True, save_trace=False)
    res.about = clone_debug(res.about, objmap)
    d = dbg.__dict__.copy()
    del d["_obj"]
    del d["about"]
    res.__dict__.update(d)
    return res


@contextmanager
def debug_inherit(**kwargs):
    """Context manager to automatically set attributes on DebugInfo.

    >>> with debug_inherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3
    """
    info = DebugInfo(**kwargs)
    _stack.push(info)
    try:
        yield
    finally:
        assert current_info() is info
        _stack.pop()


@contextmanager
def about(parent, relation, **kwargs):
    """Create a context manager for new DebugInfo to have a given parent and relation.

    Any DebugInfo instance created within the context manager will have its about field
    set to the parent DebugInfo (or parent.debug, if parent is not a DebugInfo), and its
    relation field set to the given relation.
    """
    parent = getattr(parent, "debug", parent)
    if _debug.get() and not isinstance(parent, DebugInfo):
        raise TypeError("about() takes a DebugInfo or an object with debug")
    with debug_inherit(about=parent, relation=relation, **kwargs):
        yield


def attach_debug_info(obj, **kwargs):
    """Attach a new DebugInfo to the object with the provided attributes.

    Returns:
        The object, with its debug field set to a new DebugInfo.
    """
    info = DebugInfo(obj, **kwargs)
    obj.debug = info
    return obj


class Labeler:
    """Labeling system for DebugInfo.

    Arguments:
        reverse_order: Whether to reverse the order of relations (defaults to
            False). The parts returned by relation_translator will be reversed.
            Values:
                False: Generate something like "f:while:body"
                True: Generate something like "body:while:f"
    """

    def __init__(self, reverse_order=False):
        """Initialize the Labeler."""
        self.generated_names = Counter()
        self.cache = {}
        self.namecache = {}
        self.name_id = count(1)
        self.reverse_order = reverse_order

    def translate_relation(self, rel):
        """Default relation translator for Labeler.

        Given a relation, creates a label like "relation:", using a colon as a
        delimiter.
        """
        return [":", rel or "_"]

    def generate_name(self, id):
        """Default name generator for Labeler.

        Given an id, return a label like "#id".
        """
        return str(f"#{id}")

    def disambiguate(self, label, id):
        """Default disambiguator for Labeler.

        Given a label and an id (representing how many identical labels were
        generated for different nodes prior to this one), return a label like
        "label~id".
        """
        return str(f"{label}~{id}")

    def describe_object(self, obj):
        """Describe objects by their value.

        Return None if the object should be described by name.
        """
        return None

    def label(self, info, generate=True):
        """Generate a label for the DebugInfo.

        The label may not be unique. Multiple DebugInfo could have the same
        label.
        """
        chain = info.get_chain()
        chain.reverse()
        (first, _), *rest = chain

        rval = []

        if first not in self.namecache:
            name = first.name
            if name is None:
                if not generate:
                    return None
                name = self.generate_name(next(self.name_id))
            rval.append(name)
            self.namecache[first] = name
        else:
            rval.append(self.namecache[first])

        for info, relation in rest:
            rval.extend(self.translate_relation(relation))

        if self.reverse_order:
            rval.reverse()

        return "".join(rval)

    def __call__(self, obj, generate=True):
        """Generate a unique label for the object or DebugInfo."""
        if isinstance(obj, DebugInfo):
            info = obj
        else:
            if (lbl := self.describe_object(obj)) is not None:
                return lbl
            info = obj.debug
        key = id(obj)
        if key in self.cache:
            return self.cache[key]

        lbl = None
        if info is not None:
            lbl = self.label(info, generate=generate)
        elif generate:
            lbl = self.generate_name(next(self.name_id))
        if lbl is None:
            return None
        self.generated_names[lbl] += 1
        n = self.generated_names[lbl]
        if n > 1:
            lbl = self.disambiguate(lbl, n)

        self.cache[key] = lbl
        return lbl


class AbbrvLabeler(Labeler):
    """Labeler that uses abbreviated symbols for certain relations.

    Arguments:
        relation_map: A dict from relation name to symbol.
        reverse_order: See Labeler.
    """

    def __init__(self, relation_map, **kwargs):
        self.relation_map = relation_map
        super().__init__(**kwargs)

    def translate_relation(self, rel):
        """Generate a label for the relation."""
        if rel in self.relation_map:
            return [self.relation_map[rel]]
        else:
            return super().translate_relation(rel)
