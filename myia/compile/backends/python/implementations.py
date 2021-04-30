"""Global implementations useful for compiled modules."""
from collections import Counter
from types import FunctionType

from ovld import ovld


@ovld  # noqa: F811
def myia_iter(obj: range):
    return obj


@ovld  # noqa: F811
def myia_iter(obj: tuple):
    return obj


@ovld  # noqa: F811
def myia_hasnext(obj: range):
    return obj.start < obj.stop


@ovld  # noqa: F811
def myia_hasnext(obj: tuple):
    return bool(obj)


@ovld  # noqa: F811
def myia_next(obj: range):
    return obj.start, range(obj.start + obj.step, obj.stop, obj.step)


@ovld  # noqa: F811
def myia_next(obj: tuple):
    return obj[0], obj[1:]


@ovld  # noqa: F811
def typeof(obj: object):
    return type(obj)


@ovld  # noqa: F811
def typeof(obj: FunctionType):
    # If object is a function, we return it as it is.
    return obj


class MakeHandle:
    """Helper class for apply `make_handle`."""

    def __init__(self):
        """Initialize."""
        self.counter = Counter()

    def __call__(self, object_type):
        # If object type is a function, make sure
        # to always generate the same handle.
        if isinstance(object_type, FunctionType):
            return f"function_{id(object_type)}"
        name = object_type.__name__
        self.counter.update([name])
        count = self.counter[name]
        return f"{name}_h{count}"


class Universe:
    """Helper class for Universe."""

    def __init__(self):
        """Initialize."""
        self.universe = {}

    def setitem(self, key, value):
        """Associate a key to a value."""
        self.universe[key] = value

    def getitem(self, key):
        """Return value associated to given key."""
        return self.universe[key]
