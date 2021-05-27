"""Global implementations useful for compiled modules."""
from ovld import ovld

#####################
# Iterator protocol #
#####################


@ovld
def myia_iter(obj: range):
    """Create an iterator."""
    return obj


@ovld
def myia_iter(obj: tuple):  # noqa: F811
    return obj


@ovld
def myia_hasnext(obj: range):
    """Return whether the iterator has a next production."""
    return obj.start < obj.stop


@ovld
def myia_hasnext(obj: tuple):  # noqa: F811
    return bool(obj)


@ovld
def myia_next(obj: range):
    """Return the iterator's next production and the next iterator."""
    return obj.start, range(obj.start + obj.step, obj.stop, obj.step)


@ovld
def myia_next(obj: tuple):  # noqa: F811
    return obj[0], obj[1:]


##############################
# Handle and global universe #
##############################


class Handle:
    """Handle class for `make_handle`."""

    def __init__(self, object_type, value=None):
        self.type = object_type
        self.value = value


def make_handle(object_type):
    """Create a handle of the given type."""
    return Handle(object_type)


def global_universe_getitem(h):
    """Set the value of a handle."""
    return h.value


def global_universe_setitem(h, value):
    """Get the value of a handle."""
    h.value = value


########
# Misc #
########


def apply(fn, *groups):
    """Apply fn to the given arguments.

    Each group must be either a tuple of arguments or a dictionary of
    keyword arguments.
    """
    args = []
    kwargs = {}
    for g in groups:
        if isinstance(g, tuple):
            args.extend(g)
        else:
            kwargs.update(g)
    return fn(*args, **kwargs)


def make_tuple(*args):
    """Make a tuple."""
    return args


def make_list(*args):
    """Make a list."""
    return list(args)


def make_dict(*args):
    """Make a dict."""
    res = {}
    for i in range(len(args) // 2):
        res[args[2 * i]] = args[2 * i + 1]
    return res


def switch(cond, if_true, if_false):
    """Return if_true if cond is True, else return if_false."""
    return if_true if cond else if_false


def user_switch(cond, if_true, if_false):
    """Return if_true if cond is True, else return if_false."""
    return if_true if cond else if_false


def raise_(exc):
    """Raise an exception."""
    raise exc


def return_(x):
    """Return the input."""
    return x


def resolve(ns, key):
    """Resolve the name in the given namespace."""
    return ns[key]
