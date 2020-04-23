"""Lambda lifting."""

from types import SimpleNamespace as NS

from ..info import About
from ..ir import ANFNode, manage
from ..utils import WorkSet
from ..utils.errors import untested_legacy


def _find_fvs(graph):
    fvs = set()
    work = WorkSet([graph])
    for g in work:
        for fv in g.free_variables_total:
            if isinstance(fv, ANFNode):
                fvs.add(fv)
            else:
                work.queue(fv)
    return [fv for fv in fvs if fv.graph not in graph.scope]


def lambda_lift(root):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all calls will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.
    """
    mng = manage(root)
    graphs = WorkSet(mng.graphs)
    todo = {}

    for g in graphs:
        if g.parent and not graphs.processed(g.parent):
            with untested_legacy():
                # The manager seems to naturally sort graphs so that parents are
                # before their children but I don't think we have any guarantee
                graphs.requeue(g)
                continue

        todo[g] = NS(graph=g, calls=[], fvs={})

        if not g.free_variables_total:
            continue
        gcalls = [
            (node, idx)
            for ct in mng.graph_constants[g]
            for node, idx in mng.uses[ct]
        ]
        if all(idx == 0 for node, idx in gcalls):

            def _param(fv):
                with About(fv.debug, "llift"):
                    param = g.add_parameter()
                    param.abstract = fv.abstract
                    return param

            todo[g] = NS(
                graph=g,
                calls=gcalls,
                fvs={fv: _param(fv) for fv in _find_fvs(g)},
            )

    with mng.transact() as tr:
        for entry in todo.values():
            # Add the new parameters to all call sites
            for node, _ in entry.calls:
                new_node = node.graph.apply(*node.inputs, *entry.fvs)
                new_node.abstract = node.abstract
                tr.replace(node, new_node)

    with mng.transact() as tr:
        for entry in todo.values():
            # Redirect the fvs to the parameter (those in scope)
            for fv, param in entry.fvs.items():
                for node, idx in mng.uses[fv]:
                    if node.graph in entry.graph.scope:
                        tr.set_edge(node, idx, param)

    return root


__consolidate__ = True
__all__ = ["lambda_lift"]
