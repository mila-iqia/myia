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
