"""Closure conversion."""

from collections import defaultdict

from ..info import About
from ..ir import Constant, Graph, Parameter, manage, sexp_to_node
from ..operations import primitives as P


def closure_convert(root):
    """Closure-convert all graphs starting from root.

    The resulting graphs will have no free variables, but will instead get the
    values of their free variables through additional arguments placed at the
    beginning.

    This is a destructive operation.
    """
    mng = manage(root)
    fvs = {gg: list(g_fvs) for gg, g_fvs in mng.free_variables_total.items()}

    with mng.transact() as tr:
        repl = defaultdict(dict)
        for g in mng.graphs:
            new_params = []
            for node in fvs.get(g, []):
                with About(node.debug, "fv"):
                    param = Parameter(g)
                    param.abstract = node.abstract
                    new_params.append(param)
                tr.set_parameters(g, new_params + g.parameters)
                repl[g][node] = param

        closures = [(g, g.parent) for g in mng.graphs if g.parent]

        for g, parent in closures:
            # This loop creates an incomplete partial() call and sets it in the
            # repl directory immediately.
            sexp = (P.partial, g)
            repl[parent][g] = sexp_to_node(sexp, parent)

        for g, parent in closures:
            # This loop completes the partials. It's important to do this using
            # two loops, because a closure's free variables may contain a
            # different partial, so we want all of them to be available.
            closure_args = []
            for fv in fvs[g]:
                if isinstance(fv, Graph):
                    arg = repl[parent].get(fv, Constant(fv))
                else:
                    arg = repl[parent].get(fv, fv)
                closure_args.append(arg)
            repl[parent][g].inputs[2:] = closure_args

        for g in mng.graphs:
            rg = repl[g]
            for node in g.nodes:
                if node.is_apply():
                    for i, inp in enumerate(node.inputs):
                        if inp in rg:
                            tr.set_edge(node, i, rg[inp])
                        elif inp.is_constant_graph() and inp.value in rg:
                            tr.set_edge(node, i, rg[inp.value])

    return root


__consolidate__ = True
__all__ = ["closure_convert"]
