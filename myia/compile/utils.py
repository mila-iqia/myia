"""Utility functions for graph compilation and code generation."""
from dataclasses import dataclass

from ..abstract import AbstractValue
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
        from ..pipeline.steps import convert_result
        res = self.backend.to_value(self.value, self.vm_t)
        return convert_result(res, self.orig_t, self.vm_t)


def get_outputs(lst, uses, seen):
    """Return the list of nodes whose values are required beyond this segment.

    Arguments:
        lst: list of nodes (the segment)
        uses: dict mapping each node to its uses (globally)
        seen: set of nodes that are part of the segment

    """
    outputs = []
    for n in lst:
        if n.is_apply() and any(u[0] not in seen for u in uses[n]):
            outputs.append(n)
    return outputs
