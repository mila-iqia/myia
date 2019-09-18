"""Classes to support universes."""

from dataclasses import dataclass

from .serialize import serializable


@serializable('Universe')
class UniverseInstance:
    """Universe mapping references to values.

    Keys are HandleInstances, which contain the expected abstract type.
    """

    def __init__(self, _contents={}):
        """Initialize a UniverseInstance."""
        self._contents = dict(_contents)

    def _serialize(self):
        return self._contents

    @classmethod
    def _construct(cls):
        res = cls([])
        data = yield res
        res._contents.update(data)

    def get(self, handle):
        """Get the value associated to the handle."""
        if handle not in self._contents:
            self._contents[handle] = handle.initial
        return self._contents[handle]

    def set(self, handle, value):
        """Set a value for the given handle."""
        rval = UniverseInstance(self._contents)
        rval._contents[handle] = value
        return rval


@serializable('Handle')
@dataclass(frozen=True, eq=False)
class HandleInstance:
    """Key to use in an Universe."""

    initial: object


new_universe = UniverseInstance()


__all__ = [
    "UniverseInstance",
    "HandleInstance",
    "new_universe",
]
