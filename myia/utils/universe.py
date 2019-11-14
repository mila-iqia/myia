"""Classes to support universes."""
from . import serializable


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
        res = cls()
        data = yield res
        res._contents = data

    def get(self, handle):
        """Get the value associated to the handle."""
        if handle in self._contents:
            return self._contents[handle]
        else:
            return handle.state

    def set(self, handle, value):
        """Set a value for the given handle."""
        rval = UniverseInstance(self._contents)
        rval._contents[handle] = value
        return rval

    def commit(self):
        """Change the state of all handles to their corresponding values."""
        for handle, value in self._contents.items():
            handle.state = value


class HandleInstance:
    """Key to use in an Universe."""

    def __init__(self, state):
        """Initialize a HandleInstance."""
        self.state = state


new_universe = UniverseInstance()


__all__ = [
    "UniverseInstance",
    "HandleInstance",
    "new_universe",
]
