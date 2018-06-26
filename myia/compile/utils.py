"""Utility functions for graph compilation and code generation."""

from ..ir import is_apply


def get_outputs(lst, uses, seen):
    """Return the list of nodes whose values are required beyond this segment.

    Arguments:
        lst: list of nodes (the segment)
        uses: dict mapping each node to its uses (globally)
        seen: set of nodes that are part of the segment

    """
    outputs = []
    for n in lst:
        if is_apply(n) and any(u[0] not in seen for u in uses[n]):
            outputs.append(n)
    return outputs
