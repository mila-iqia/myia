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
    yield
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

    def __new__(cls, *args, **kwargs):
        if not _debug.get():
            return None
        else:
            return super().__new__(cls)

    def __init__(self, obj=None, **kwargs):
        """Construct a DebugInfo object."""
        self.name = None
        self.about = None
        self.relation = None
        self.save_trace = False
        self.trace = None
        self._obj = None

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
    yield
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


def default_relation_translator(rel):
    """Default relation translator for Labeler.

    Given a relation, creates a label like "relation:", using a colon as a
    delimiter.
    """
    return f"{rel or '_'}:"


def default_name_generator(id):
    """Default name generator for Labeler.

    Given an id, return a label like "#id".
    """
    return str(f"#{id}")


def default_disambiguator(label, id):
    """Default disambiguator for Labeler.

    Given a label and an id (representing how many identical labels were
    generated for different nodes prior to this one), return a label like
    "label//id".
    """
    return str(f"{label}//{id}")


def default_object_describer(obj):
    """Default function to describe objects in Labeler.

    Always returns None.
    """
    return None


class AbbreviationTranslator:
    """Relation translator that uses abbreviated symbols for certain relations.

    Arguments:
        relation_map: A dict from relation name to symbol.
        default_translator: Translator to use for relations that are not in
            the relation_map.
    """

    def __init__(
        self, relation_map, default_translator=default_relation_translator
    ):
        """Initialize the AbbreviationTranslator."""
        self.relation_map = relation_map
        self.default_translator = default_translator

    def __call__(self, relation):
        """Generate a label for the relation."""
        if relation in self.relation_map:
            return self.relation_map[relation]
        else:
            return self.default_translator(relation)


class Labeler:
    """Labeling system for DebugInfo.

    Arguments:
        relation_translator: Function from a relation string to a label part.
        name_generator: Function to generate a fresh name from a numeric id.
        disambiguator: Function to disambiguate duplicate labels. Takes the
            label and a number (monotonically increasing for each duplicate.)
        object_describer: Function to describe a non-DebugInfo object. If the
            function returns None, the DebugInfo will be labeled instead. The
            default object_describer always returns None.
    """

    def __init__(
        self,
        relation_translator=default_relation_translator,
        name_generator=default_name_generator,
        disambiguator=default_disambiguator,
        object_describer=default_object_describer,
    ):
        """Initialize the Labeler."""
        self.generated_names = Counter()
        self.cache = {}
        self.namecache = {}
        self.name_id = count(1)
        self.relation_translator = relation_translator
        self.name_generator = name_generator
        self.disambiguator = disambiguator
        self.object_describer = object_describer

    def label(self, info):
        """Generate a label for the DebugInfo.

        The label may not be unique. Multiple DebugInfo could have the same
        label.
        """
        chain = info.get_chain()
        first = chain[-1][0]
        if first not in self.namecache:
            rval = first.name
            if rval is None:
                rval = self.name_generator(next(self.name_id))
            self.namecache[first] = rval
        else:
            rval = self.namecache[first]

        for info, relation in chain[:-1]:
            rval = f"{self.relation_translator(relation)}{rval}"
        return rval

    def __call__(self, obj):
        """Generate a unique label for the object or DebugInfo."""
        if isinstance(obj, DebugInfo):
            info = obj
        else:
            if (lbl := self.object_describer(obj)) is not None:
                return lbl
            info = obj.debug
        key = id(obj)
        if key in self.cache:
            return self.cache[key]

        if info is not None:
            lbl = self.label(info)
        else:
            lbl = self.name_generator(next(self.name_id))
        self.generated_names[lbl] += 1
        n = self.generated_names[lbl]
        if n > 1:
            lbl = self.disambiguator(lbl, n)

        self.cache[key] = lbl
        return lbl
