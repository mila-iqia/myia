"""Global implementations useful for compiled modules."""
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


class Handle:
    """Handle class for `make_handle`."""

    def __init__(self, object_type):
        self.type = object_type


class Universe:
    """Helper class for Universe."""

    def __init__(self):
        self.universe = {}

    def setitem(self, key, value):
        """Associate a key to a value."""
        self.universe[key] = value

    def getitem(self, key):
        """Return value associated to given key."""
        return self.universe[key]
