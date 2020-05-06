"""Lambda lifting."""

from types import SimpleNamespace as NS

from ..info import About
from ..ir import Graph, manage
from ..utils import WorkSet


def lambda_lift(root):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all calls will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.
    """
    mng = manage(root)
    graphs = WorkSet(mng.graphs)
    candidates = []

    # Step 1a: Figure out what to do. We will try to lift all functions that
    # have free variables and are only used in call position.
    for g in graphs:
        if g.parent and not graphs.processed(g.parent):
            # Process parents first (we will reverse the order later)
            graphs.requeue(g)
            continue

        if g.free_variables_total and g.all_direct_calls:
            candidates.append(g)

    # Step 1b: We try to complete the scope of each candidate with the graphs
    # that are free variables for that candidate. If they are also candidates,
    # there is nothing more to do, but if they are not candidates, they must
    # be moved inside the scope. We only do this if all uses for the graph
    # that's a free variable of the candidate is in the genuine scope of the
    # candidate.
    todo = []
    for g in candidates:

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

        todo.append(
            NS(
                graph=g,
                calls=g.call_sites,
                fvs={fv: _param(fv) for fv in g.free_variables_extended},
                scope={*g.scope, *fvg},
            )
        )

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
            for fv, param in entry.fvs.items():
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


__consolidate__ = True
__all__ = ["lambda_lift"]
