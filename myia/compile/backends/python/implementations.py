"""Global implementations useful for compiled modules."""
from collections import Counter

from ovld import ovld


@ovld
def myia_iter(obj: range):
    return obj


@ovld
def myia_iter(obj: tuple):  # noqa: F811
    return obj


@ovld
def myia_hasnext(obj: range):
    return obj.start < obj.stop


@ovld
def myia_hasnext(obj: tuple):  # noqa: F811
    return bool(obj)


@ovld
def myia_next(obj: range):
    return obj.start, range(obj.start + obj.step, obj.stop, obj.step)


@ovld
def myia_next(obj: tuple):  # noqa: F811
    return obj[0], obj[1:]


def typeof(obj: object):
    """Implementation for apply `typeof`."""
    return type(obj)


class MakeHandle:
    """Helper class for apply `make_handle`."""

    def __init__(self):
        """Initialize."""
        self.counter = Counter()

    def __call__(self, object_type):
        """Generate an handle for given object type."""
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
