"""Global implementations useful for compiled modules."""
from ovld import ovld


#####################
# Iterator protocol #
#####################


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


##############################
# Handle and global universe #
##############################


class Handle:
    """Handle class for `make_handle`."""

    def __init__(self, object_type):
        self.type = object_type
        self.value = None


def make_handle(object_type):
    return Handle(object_type)


def global_universe_getitem(h):
    return h.value


def global_universe_setitem(h, value):
    h.value = value


########
# Misc #
########


def apply(fn, *groups):
    args = []
    kwargs = {}
    for g in groups:
        if isinstance(g, tuple):
            args.extend(g)
        else:
            kwargs.update(g)
    return fn(*args, **kwargs)


def make_tuple(*args):
    return args


def make_list(*args):
    return list(args)


def make_dict(*args):
    res = {}
    for i in range(len(args) // 2):
        res[args[2 * i]] = args[2 * i + 1]
    return res


def switch(cond, if_true, if_false):
    return if_true if cond else if_false


def user_switch(cond, if_true, if_false):
    return if_true if cond else if_false


def raise_(exc):
    raise exc


def return_(x):
    return x


def resolve(ns, key):
    return ns[key]
