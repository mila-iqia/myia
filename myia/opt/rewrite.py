"""Optimizations that rewrite graph interfaces.

All it currently does is eliminate unused arguments in functions.
"""

from ..utils import Partializable, tracer


def rewrite_graphs(root, manager):
    """Remove unused parameters in graphs whenever possible.

    * The root graph's parameters are not touched
    * Only graphs that only have direct calls (not used in HOFs) are processed
    """
    changes = False
    for g in list(manager.graphs):
        if g not in manager.graphs:
            # The graph might have been dropped if the only reference was an
            # input that has been dropped.
            continue
        if not g.all_direct_calls:
            continue

        # Drop unused inputs
        keep = [bool(manager.uses[param]) for param in g.parameters]
        if g is not root and not all(keep):
            with manager.transact() as tr:
                tr.set_parameters(
                    g, [p for p, keep in zip(g.parameters, keep) if keep]
                )
                for call in g.call_sites:
                    new_call = call.graph.apply(
                        call.inputs[0],
                        *[
                            arg
                            for arg, keep in zip(call.inputs[1:], keep)
                            if keep
                        ]
                    )
                    new_call.abstract = call.abstract
                    tr.replace(call, new_call)
            changes = True
    return changes


class RewriteGraphs(Partializable):
    """Remove unused arguments etc."""

    def __init__(self, resources):
        """Initialize RewriteGraphs."""
        self.resources = resources
        self.name = "rewrite_graphs"

    def __call__(self, root):
        """Apply rewrite_graphs on root."""
        args = dict(
            opt=self,
            node=None,
            manager=self.resources.opt_manager,
            profile=False,
        )
        with tracer("opt", **args) as tr:
            tr.set_results(success=False, **args)
            chg = rewrite_graphs(root, self.resources.opt_manager)
            if chg:
                tracer().emit_success(**args, new_node=None)
            return chg


__all__ = ["RewriteGraphs", "rewrite_graphs"]
