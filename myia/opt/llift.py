"""Lambda lifting."""

from types import SimpleNamespace as NS

from ..info import About
from ..ir import Graph, manage
from ..operations import primitives as P
from ..utils import OrderedSet, WorkSet


def _get_switch_sibling(g):
    """Returns the other graph of a switch statement.

    Given g, return (x, g2) if g is only used in an expression of the form
    `x = switch(cond, g, g2)`. Otherwise, return None, None.
    """
    hos = g.higher_order_sites
    if g.call_sites or len(hos) != 1:
        return None, None
    ((site, key),) = hos
    if site.is_apply(P.switch) and (key == 2 or key == 3):
        g2 = site.inputs[5 - key]
        if g2.is_constant_graph():
            return site, g2.value
    return None, None


def lambda_lift(root, lift_switch=True):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all calls will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.

    Arguments:
        root: The root graph from which to proceed.
        lift_switch: If True, expressions of the form
            switch(test, f, g)(...) may lead to f and g to be lifted
            if f and g are only used in the switch statement.
    """
    mng = manage(root)
    mng.gc()
    graphs = WorkSet(mng.graphs)
    candidates = {}

    # Step 1a: Figure out what to do. We will try to lift all functions that
    # have free variables and are only used in call position.
    for g in graphs:
        if g in candidates:
            continue

        if g.parent and not graphs.processed(g.parent):
            # Process parents first (we will reverse the order later)
            graphs.requeue(g)
            continue

        if g.free_variables_total and g.all_direct_calls:
            candidates[g] = NS(
                graph=g, calls=g.call_sites, fvs=g.free_variables_extended,
            )

        if lift_switch:
            switch_node, g2 = _get_switch_sibling(g)
            if g2 and _get_switch_sibling(g2)[1] is g:
                switch_uses = mng.uses[switch_node]
                if len(switch_uses) == 1:
                    ((switch_caller, key),) = switch_uses
                    if key == 0:
                        all_fvs = list(
                            OrderedSet(
                                [
                                    *g.free_variables_extended,
                                    *g2.free_variables_extended,
                                ]
                            )
                        )
                        candidates[g] = NS(
                            graph=g, calls=[switch_caller], fvs=all_fvs,
                        )
                        # switch_caller is already in the calls list for g,
                        # so we don't include it for g2
                        candidates[g2] = NS(graph=g2, calls=[], fvs=all_fvs,)

    # Step 1b: We try to complete the scope of each candidate with the graphs
    # that are free variables for that candidate. If they are also candidates,
    # there is nothing more to do, but if they are not candidates, they must
    # be moved inside the scope. We only do this if all uses for the graph
    # that's a free variable of the candidate is in the genuine scope of the
    # candidate.
    todo = []
    for g, entry in candidates.items():

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
            continue

        entry.scope = {*g.scope, *fvg}
        entry.new_params = {fv: _param(fv) for fv in entry.fvs}
        todo.append(entry)

    # Step 1c: Reverse the order so that children are processed before parents.
    # This is important for step 3, because children that are lambda lifted
    # must replace their uses first (otherwise the original fvs would be
    # replaced by their parent's parameters, which is not what we want)
    todo.reverse()

    # Step 2: Add arguments to call sites.
    # For each closure, we add arguments to each call of the closure. The
    # arguments that are added are the original free variables.
    for entry in todo:
        with mng.transact() as tr:
            for node in entry.calls:
                new_node = node.graph.apply(*node.inputs, *entry.fvs)
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


###########################
# If llift_switch is True #
###########################

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
