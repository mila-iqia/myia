"""Common subexpression elimination."""


from collections import defaultdict

from ..graph_utils import toposort
from ..ir import succ_incoming
from ..utils import Partializable


def cse(root, manager):
    """Apply CSE on root."""
    hashes = {}
    groups = defaultdict(list)
    manager.add_graph(root)
    changes = False

    for g in manager.graphs:
        for node in toposort(g.return_, succ_incoming):
            if node in hashes:
                continue

            if node.is_constant():
                h = hash((node.value, node.type))
            elif node.is_apply():
                h = hash(tuple(hashes[inp] for inp in node.inputs))
            elif node.is_parameter():
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
            if main.graph is not other.graph:
                # This could happen because of a hash collision
                continue  # pragma: no cover

            elif main.is_constant() and other.is_constant():
                repl = main.type is other.type and main.value == other.value

            elif main.is_apply() and other.is_apply():
                # The inputs to both should have been merged beforehand
                # because groups is topologically sorted
                in1 = main.inputs
                in2 = other.inputs
                repl = len(in1) == len(in2) \
                    and all(i1 is i2 for i1, i2 in zip(in1, in2))

            if repl:
                changes = True
                manager.replace(other, main)

    return changes


class CSE(Partializable):
    """Common subexpression elimination."""

    def __init__(self, optimizer, report_changes=True):
        """Initialize CSE."""
        self.optimizer = optimizer
        self.report_changes = report_changes

    def __call__(self, root):
        """Apply CSE on root."""
        chg = cse(root, self.optimizer.resources.manager)
        return chg and self.report_changes
