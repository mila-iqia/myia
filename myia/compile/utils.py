"""Utility functions for graph compilation and code generation."""
from dataclasses import dataclass

from ..abstract import AbstractValue, to_abstract
from .backends import Backend


@dataclass(frozen=True)
class BackendValue:
    """Class that represents a value in a backend."""

    value: object
    orig_t: AbstractValue
    vm_t: AbstractValue
    backend: Backend

    def from_device(self):
        """Get a python value for this backend value."""
        from ..simplify_types import from_canonical
        res = self.backend.from_backend_value(self.value, self.vm_t)
        return from_canonical(res, self.orig_t)

    def __getattr__(self, name):  # pragma: no cover
        raise AttributeError(
            "You attempted to get an attribute on a BackendValue. This is "
            "most likely an error. If you want access to the python object, "
            "call .from_device() on this object.")


@to_abstract.register
def _to_abstract(self, v: BackendValue, **kwargs):
    return v.orig_t


__all__ = [
    'BackendValue',
]
