"""Common subexpression elimination."""


from collections import defaultdict

from ..graph_utils import toposort
from ..ir import succ_incoming, is_constant, is_apply, is_parameter


def cse(root, manager):
    """Apply CSE on root."""
    hashes = {}
    groups = defaultdict(list)
    manager.add_graph(root)

    for g in manager.graphs:
        for node in toposort(g.return_, succ_incoming):
            if node in hashes:
                continue

            if is_constant(node):
                h = hash((node.value, type(node.value)))
            elif is_apply(node):
                h = hash(tuple(hashes[inp] for inp in node.inputs))
            elif is_parameter(node):
                h = hash(node)
            else:
                raise TypeError(f'Unknown node type: {node}') \
                    # pragma: no cover

            hashes[node] = h
            groups[h].append(node)

    # Note: this relies on dict keeping insertion order, so that the groups
    # dict is basically topologically ordered.

    for h, group in groups.items():
        main, *others = group
        for other in others:
            assert main.graph is other.graph
            if is_constant(main) and is_constant(other):
                v1 = main.value
                v2 = other.value
                repl = type(v1) is type(v2) and v1 == v2
            elif is_apply(main) and is_apply(other):
                # The inputs to both should have been merged beforehand
                # because groups is topologically sorted
                in1 = main.inputs
                in2 = other.inputs
                repl = len(in1) == len(in2) \
                    and all(i1 is i2 for i1, i2 in zip(in1, in2))
            else:
                raise AssertionError('CSE put together incompatible nodes.')

            if repl:
                manager.replace(other, main)

    return root


class CSE:
    """Common subexpression elimination."""

    def __init__(self, optimizer):
        """Initialize CSE."""
        self.optimizer = optimizer

    def __call__(self, root):
        """Apply CSE on root."""
        cse(root, self.optimizer.resources.manager)
