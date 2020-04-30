"""Lambda lifting."""

from types import SimpleNamespace as NS

from ..info import About
from ..ir import manage
from ..utils import OrderedSet, WorkSet


def lambda_lift(root):
    """Lambda-lift the graphs that can be lifted.

    Graphs with free variables for which we can identify all calls will be
    modified to take these free variables as extra arguments.

    This is a destructive operation.
    """
    mng = manage(root)
    graphs = WorkSet(mng.graphs)
    todo = []

    # Step 1: Figure out what to do
    # For each graph that is a closure, we collect all uses. If all uses are in
    # call position, this means we can easily modify the function's interface,
    # so we collect all the free variables and create new parameters for them.
    for g in graphs:
        if g.parent and not graphs.processed(g.parent):
            # Process parents first (we will reverse the order later)
            graphs.requeue(g)
            continue

        if not g.free_variables_total:
            continue

        if not g.all_direct_calls:
            continue

        def _param(fv):
            with About(fv.debug, "llift"):
                param = g.add_parameter()
                param.abstract = fv.abstract
                return param

        todo.append(
            NS(
                graph=g,
                calls=g.call_sites,
                fvs={fv: _param(fv) for fv in g.free_variables_extended},
            )
        )

    # Reverse the order so that children are processed before parents. This is
    # important for step 3, because children that are lambda lifted must
    # replace their uses first (otherwise the original fvs would be replaced
    # by their parent's parameters, which is not what we want)
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
                    if node.graph in entry.graph.scope:
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
