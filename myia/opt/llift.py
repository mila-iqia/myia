"""Lambda lifting."""

from collections import defaultdict, deque
from types import SimpleNamespace as NS

from ..info import About
from ..ir import ANFNode, Constant, Graph, Parameter, manage, sexp_to_node
from ..operations import primitives as P


def _find_fvs(graph):
    fvs = set()
    processed = set()
    graphs = deque([graph])
    while graphs:
        g = graphs.pop()
        if g in processed:
            continue
        processed.add(g)
        for fv in g.free_variables_total:
            if isinstance(fv, ANFNode):
                fvs.add(fv)
            else:
                graphs.append(fv)
    return [fv for fv in fvs if fv.graph not in graph.scope]


def lambda_lift(root):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all calls will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.
    """
    mng = manage(root)
    processed = set()
    graphs = deque(mng.graphs)
    todo = {}

    while graphs:
        g = graphs.popleft()
        if g.parent and g.parent not in processed:
            graphs.push(g)
            continue
        if g in processed:
            continue

        processed.add(g)
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
