"""Utilities to merge dictionaries and other data structures."""


from ovld import ovld

from .misc import MISSING, Named, Registry

# Use in a merge to indicate that a key should be deleted
DELETE = Named("DELETE")


class MergeMode:
    """Wraps a value to control how it is merged.

    Class attributes:
        mode: The merge mode to use.

    Attributes:
        value: The value to merge.

    """

    mode = "merge"  # NOTE: This is the default merge mode used by merge()

    def __init__(self, __value=MISSING, **kwargs):
        """Initialize a MergeMode.

        The value to merge will be the single argument given, or if there is no
        positional argument, the keywords dictionary.
        """
        if __value is MISSING:
            self.value = kwargs
        else:
            assert not kwargs
            self.value = __value


class Merge(MergeMode):
    """Merge normally."""

    mode = "merge"


class Override(MergeMode):
    """Do not concatenate sequences."""

    mode = "override"


class Reset(MergeMode):
    """Throw away the previous value."""

    mode = "reset"


###########
# cleanup #
###########


@ovld
def cleanup(value: object):
    return value


@ovld  # noqa: F811
def cleanup(mm: MergeMode):
    return mm.value


@ovld  # noqa: F811
def cleanup(d: dict):
    return type(d)({k: cleanup(v) for k, v in d.items() if v is not DELETE})


@ovld  # noqa: F811
def cleanup(xs: (tuple, list, set)):
    return type(xs)(cleanup(x) for x in xs)


#########
# merge #
#########


@ovld.dispatch
def merge(self, a, b, mode=MergeMode.mode):
    """Merge two data structures.

    Arguments:
        a: The original data.
        b: The data to merge.
        mode ({'merge', 'override', 'reset'}):

            :'merge': (default) Sequences will be concatenated, sets merged,
                and dictionaries merged according to their keys.
            :'override': Dictionaries are merged, but sequences are not
                concatenated.
            :'reset': b is returned, or takes primacy over the data in a.

    """
    if isinstance(b, MergeMode):
        mode = b.mode
        b = b.value
    assert not isinstance(a, MergeMode)
    return self.resolve(a, b, mode)(a, b, mode)


@ovld  # noqa: F811
def merge(d1: dict, d2, mode):
    if mode == "reset":
        assert not isinstance(d1, Registry)
        return type(d1)(d2)

    if isinstance(d1, Registry):
        rval = Registry(default_field=d1.default_field)
    else:
        rval = type(d1)()
    for k, v in d1.items():
        if k in d2:
            v2 = d2[k]
            if v2 is DELETE:
                pass
            else:
                rval[k] = merge(v, v2, mode)
        else:
            rval[k] = v
    for k, v in d2.items():
        if k not in d1:
            rval[k] = cleanup(v)
    return rval


@ovld  # noqa: F811
def merge(xs: tuple, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == "merge":
        return xs + ys
    else:
        return ys


@ovld  # noqa: F811
def merge(xs: list, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == "merge":
        return xs + ys
    else:
        return ys


@ovld  # noqa: F811
def merge(xs: set, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == "merge":
        return xs | ys
    else:
        return ys


@ovld  # noqa: F811
def merge(a: object, b, mode):
    if hasattr(a, "__merge__"):
        return a.__merge__(b, mode)
    else:
        return cleanup(b)


__consolidate__ = True
__all__ = [
    "DELETE",
    "Merge",
    "MergeMode",
    "Override",
    "Reset",
    "cleanup",
    "merge",
]
