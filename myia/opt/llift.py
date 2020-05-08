"""Lambda lifting."""

import operator
from functools import reduce
from types import SimpleNamespace as NS

from .dde import make_dead
from ..info import About
from ..ir import Constant, Graph, manage
from ..operations import primitives as P
from ..utils import OrderedSet, WorkSet


def _get_call_sites(g, mng):
    """Returns (call_sites, eqv).

    A call site C is either:

    * C = g(...)
      * Contributes nothing to eqv
    * C = switch(cond, g, g2)(...)
      * Contributes g2 to eqv
    """

    call_sites = list(g.call_sites)
    eqv = OrderedSet([g])

    for node, key in g.higher_order_sites:
        if not (node.is_apply(P.switch) and (key == 2 or key == 3)):
            return None, None

        g2_node = node.inputs[5 - key]
        if not g2_node.is_constant_graph():
            return None, None

        g2 = g2_node.value
        uses = mng.uses[node]
        if not all(key2 == 0 for site, key2 in uses):
            return None, None

        call_sites += [site for site, _ in uses]
        eqv.add(g2)

    return call_sites, eqv


def lambda_lift(root):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all call sites will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.

    Arguments:
        root: The root graph from which to proceed.
    """
    mng = manage(root)
    mng.gc()
    candidates = {}
    # Graphs that may be called at a given call site
    call_site_poss = {}

    # Step 1a: Figure out what to do. We will try to lift all functions that
    # have free variables and are only used in call position, or as a branch
    # of a switch only used in call position, in which case we will merge
    # the free variables needed by all branches.
    for g in mng.graphs:
        if g in candidates:
            continue

        valid = True
        ws = WorkSet([g])
        # Set of graphs that must all be augmented with the same fvs
        eqv = OrderedSet()
        # Call sites that will have to be updated
        call_sites = OrderedSet()
        for currg in ws:
            eqv.add(currg)
            new_call_sites, new_eqv = _get_call_sites(currg, mng)
            if new_call_sites is None:
                valid = False
                break
            for cs in new_call_sites:
                call_site_poss[cs] = new_eqv
            call_sites.update(new_call_sites)
            ws.queue_all(new_eqv)

        if valid and any(gg.free_variables_total for gg in eqv):
            all_fvs = reduce(
                operator.or_,
                [gg.free_variables_extended for gg in eqv],
                OrderedSet(),
            )
            for gg in eqv:
                candidates[gg] = NS(
                    graph=g, calls=call_sites, fvs=all_fvs, eqv=eqv,
                )
                call_sites = []

    # Step 1b: We try to complete the scope of each candidate with the graphs
    # that are free variables for that candidate. If they are also candidates,
    # there is nothing more to do, but if they are not candidates, they must
    # be moved inside the scope. We only do this if all uses for the graph
    # that's a free variable of the candidate is in the genuine scope of the
    # candidate.
    for g, entry in list(candidates.items()):

        def _param(fv):
            with About(fv.debug, "llift"):
                param = g.add_parameter()
                param.abstract = fv.abstract
                return param

        fvg = {
            g2
            for g2 in g.free_variables_total
            if isinstance(g2, Graph) and g2 not in candidates
        }

        if not all(
            all(user in g.scope for user in g2.graph_users) for g2 in fvg
        ):
            for g2 in entry.eqv:
                del candidates[g2]
            continue

        entry.scope = {*g.scope, *fvg}
        entry.new_params = {fv: _param(fv) for fv in entry.fvs}

    # Step 1c: Reverse the order so that children are processed before parents.
    # This is important for step 3, because children that are lambda lifted
    # must replace their uses first (otherwise the original fvs would be
    # replaced by their parent's parameters, which is not what we want)
    todo = []
    ws = WorkSet(candidates.keys())
    for g in ws:
        if g.parent and g.parent in candidates and not ws.processed(g.parent):
            # Add parents first (we will reverse the order later)
            ws.requeue(g)
            continue
        todo.append(candidates[g])
    todo.reverse()

    # Step 2: Add arguments to call sites.
    # For each closure, we add arguments to each call of the closure. The
    # arguments that are added are the original free variables.
    for entry in todo:
        with mng.transact() as tr:
            for node in entry.calls:
                fvs = [
                    fv
                    if any(
                        fv in gg.free_variables_extended
                        for gg in call_site_poss[node]
                    )
                    else make_dead(fv)
                    for fv in entry.fvs
                ]
                new_node = node.graph.apply(*node.inputs, *fvs)
                new_node.abstract = node.abstract
                tr.replace(node, new_node)

    # Step 3: Redirect the free variables
    # For each closure, we redirect all free variables within scope by the
    # new parameters, which means that they are not closures anymore. This
    # may modify the arguments added in step 2. Children before parents.
    for entry in todo:
        with mng.transact() as tr:
            # Redirect the fvs to the parameter (those in scope)
            for fv, param in entry.new_params.items():
                for node, idx in mng.uses[fv]:
                    if node.graph in entry.scope:
                        tr.set_edge(node, idx, param)

    return root


###########
# EXAMPLE #
###########

##################
# ORIGINAL GRAPH #
##################

# def f(x, y):
#     def g(z):
#         return x + z

#     def h():
#         return g(y)

#     return h()

##########
# Step 1 #
##########

# def f(x, y):
#     def g(z, _x):          # <- Add parameter
#         return x + z

#     def h(_y, _x):         # <- Add parameters
#         return g(y)

#     return h()

##########
# Step 2 #
##########

# def f(x, y):
#     def g(z, _x):
#         return x + z

#     def h(_y, _x):
#         return g(y, x)     # <- Add argument to call

#     return h(y, x)         # <- Add arguments to call

##########
# Step 3 #
##########

# def f(x, y):
#     def g(z, _x):
#         return _x + z      # <- Swap fv for parameter

#     def h(_y, _x):
#         return g(_y, _x)   # <- Swap fvs for parameters

#     return h(y, x)


######################################################
# Lambda lifting when switch statements are involved #
######################################################

# def f(x, y, z):
#     def true_branch():
#         return y

#     def false_branch():
#         return z

#     return switch(x < 0, true_branch, false_branch)()

# ==>

# def f(x, y, z):
#     def true_branch(_y, _z):    # <- Add free variables of both branches
#         return _y

#     def false_branch(_y, _z):   # <- Add free variables of both branches
#         return _z

#     return switch(x < 0, true_branch, false_branch)(y, z)   # <- Add args


__consolidate__ = True
__all__ = ["lambda_lift"]
