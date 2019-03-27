"""Utilities to merge dictionaries and other data structures."""


from .misc import Named, overload


# Use in a merge to indicate that a key should be deleted
DELETE = Named('DELETE')


# Placeholder for MergeMode.__init__
_ABSENT = Named('_ABSENT')


class MergeMode:
    """Wraps a value to control how it is merged.

    Class attributes:
        mode: The merge mode to use.

    Attributes:
        value: The value to merge.

    """

    mode = 'merge'  # NOTE: This is the default merge mode used by merge()

    def __init__(self, __value=_ABSENT, **kwargs):
        """Initialize a MergeMode.

        The value to merge will be the single argument given, or if there is no
        positional argument, the keywords dictionary.
        """
        if __value is _ABSENT:
            self.value = kwargs
        else:
            assert not kwargs
            self.value = __value


class Merge(MergeMode):
    """Merge normally."""

    mode = 'merge'


class Override(MergeMode):
    """Do not concatenate sequences."""

    mode = 'override'


class Reset(MergeMode):
    """Throw away the previous value."""

    mode = 'reset'


###########
# cleanup #
###########


@overload
def cleanup(value: object):
    return value


@overload  # noqa: F811
def cleanup(mm: MergeMode):
    return mm.value


@overload  # noqa: F811
def cleanup(d: dict):
    return type(d)({k: cleanup(v) for k, v in d.items() if v is not DELETE})


@overload  # noqa: F811
def cleanup(xs: (tuple, list, set)):
    return type(xs)(cleanup(x) for x in xs)


#########
# merge #
#########


@overload.wrapper
def merge(__call__, a, b, mode=MergeMode.mode):
    """Merge two data structures.

    Arguments:
        a: The original data.
        b: The data to merge.
        mode:
            'merge': (default) Sequences will be concatenated, sets merged,
                and dictionaries merged according to their keys.
            'override': Dictionaries are merged, but sequences are not
                concatenated.
            'reset': b is returned, or takes primacy over the data in a.
    """
    if isinstance(b, MergeMode):
        mode = b.mode
        b = b.value
    assert not isinstance(a, MergeMode)
    if __call__ is None:
        if hasattr(a, '__merge__'):
            return a.__merge__(b, mode)
        else:
            return cleanup(b)
    else:
        return __call__(a, b, mode)


@overload  # noqa: F811
def merge(d1: dict, d2, mode):
    if mode == 'reset':
        return type(d1)(d2)

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


@overload  # noqa: F811
def merge(xs: tuple, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@overload  # noqa: F811
def merge(xs: list, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@overload  # noqa: F811
def merge(xs: set, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs | ys
    else:
        return ys
