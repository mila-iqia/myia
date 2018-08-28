"""Utilities to merge dictionaries and other data structures."""


from .misc import Named, TypeMap


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


_cleanup_map = TypeMap()


@_cleanup_map.register(object)
def _cleanup_object(value):
    return value


@_cleanup_map.register(MergeMode)
def _cleanup_MergeMode(mm):
    return mm.value


@_cleanup_map.register(dict)
def _cleanup_dict(d):
    return type(d)({k: cleanup(v) for k, v in d.items() if v is not DELETE})


@_cleanup_map.register(tuple, list, set)
def _cleanup_sequence(xs):
    return type(xs)(cleanup(x) for x in xs)


def cleanup(x):
    """Remove all MergeMode and DELETE instances from the data."""
    return _cleanup_map[type(x)](x)


_merge_map = TypeMap(discover=lambda cls: getattr(cls, '__merge__', None))


@_merge_map.register(dict)
def _merge_dict(d1, d2, mode):
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


@_merge_map.register(tuple)
def _merge_tuple(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@_merge_map.register(list)
def _merge_list(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs + ys
    else:
        return ys


@_merge_map.register(set)
def _merge_set(xs, ys, mode):
    xs = cleanup(xs)
    ys = cleanup(ys)
    if mode == 'merge':
        return xs | ys
    else:
        return ys


@_merge_map.register(object)
def _merge_object(a, b, mode):
    return cleanup(b)


def merge(a, b, mode=MergeMode.mode):
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
    return _merge_map[type(a)](a, b, mode)
