"""Utilities related to the Env type in Myia."""


from dataclasses import dataclass

import numpy as np
from ovld import ovld


def require_same(fns, objs):
    """Check that all objects have the same properties.

    Arguments:
        fns: A collection of functions. Each function must return the same
            result when applied to each object. For example, the functions
            may be `[type, len]`.
        objs: Objects that must be invariant with respect to the given
            functions.

    """
    o, *rest = objs
    for fn in fns:
        for obj in rest:
            if fn(o) != fn(obj):
                raise TypeError(
                    "Objects do not have the same properties:"
                    f" `{o}` and `{obj}` are not conformant."
                )


@ovld  # noqa: F811
def smap(self, arg: (list, tuple), *rest):
    seqs = [arg, *rest]
    require_same([type, len], seqs)
    return type(arg)(self(*[s[i] for s in seqs]) for i in range(len(arg)))


@ovld  # noqa: F811
def smap(self, arg: np.ndarray, *rest):
    return np.vectorize(self)(arg, *rest)


@dataclass(frozen=True)
class SymbolicKeyInstance:
    """Stores information that corresponds to a node in the graph."""

    node: "ANFNode"  # noqa: F821
    abstract: object


@smap.variant
def _add(self, x: object, y):
    return x + y


class EnvInstance:
    """Environment mapping keys to values.

    Keys are SymbolicKeyInstances, which represent nodes in the graph along
    with inferred properties.
    """

    def __init__(self, _contents={}):
        """Initialize a EnvType."""
        self._contents = dict(_contents)

    def get(self, key, default):
        """Get the sensitivity list for the given key."""
        return self._contents.get(key, default)

    def set(self, key, value):
        """Set a value for the given key."""
        rval = EnvInstance(self._contents)
        rval._contents[key] = value
        return rval

    def add(self, other):
        """Add two EnvInstances."""
        rval = EnvInstance(self._contents)
        for k, v in other._contents.items():
            v0 = rval._contents.get(k)
            if v0 is not None:
                rval._contents[k] = _add(v0, v)
            else:
                rval._contents[k] = v
        return rval

    def __len__(self):
        return len(self._contents)


newenv = EnvInstance()


__consolidate__ = True
__all__ = ["EnvInstance", "SymbolicKeyInstance", "newenv", "smap"]
