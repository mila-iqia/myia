"""Optimizations that rewrite graph interfaces."""

import operator
from functools import lru_cache, reduce
from types import SimpleNamespace as NS

from ..info import About
from ..ir import Graph, Parameter
from ..operations import primitives as P
from ..utils import OrderedSet, Partializable, WorkSet, tracer
from .dde import make_dead


def _noop(fn):
    """Mark a function to represent that the represented feature is absent."""
    fn.noop = True
    return fn


class GraphInterfaceRewriter:
    """Base class to rewrite a graph's interface.

    A graph's interface (parameter list and output type) can be rewritten if
    all of its uses are either:

    a. A direct call to the function, e.g. g(x)
    b. A branch of a switch statement that's called directly, e.g.
       switch(cond, g, h)(x)

    The run method applies the optimization and proceeds as follows:

    1. Identify "groups" of graphs that can be rewritten. If two graphs are
       different branches of the same switch statement then they must be
       rewritten identically and belong to the same group. A list of entries
       is produced at this point. Each graph is associated to one, but the
       first graph in a group gets a `calls` field in the entry that contains
       the map of the relevant call sites (others get an empty list).

    2. Filter the candidate graphs using the `filter` method, to eliminate
       those we do not wish to rewrite. Eliminating one graph in a group
       eliminates the whole group.

    3. Reorder the graphs using the `order_key` method. Some optimizations
       should be executed on parents first, or children first, for example.

    4. Rewrite the call sites using the `rewrite_call` method. The method
       may add or remove arguments as necessary, wrap or unwrap the call, etc.

    5. Rewrite the graphs using the `rewrite_graph` method, adding/removing
       parameters or changing other aspects of the interface.
    """

    relation = None

    def __init__(self, manager, graphs=None):
        """Initialize a GraphInterfaceRewriter.

        Arguments:
            manager: The manager for the graphs.
            graphs: The set of graphs to process. If not provided, it defaults
                to all the non-root graphs in the manager.
        """
        self.manager = manager
        self.graphs = (
            manager.graphs - manager.roots if graphs is None else graphs
        )

    def param(self, graph, model):
        """Create a new parameter for the graph based on the model node.

        The abstract field of the model will be copied for the new
        parameter.
        """
        with About(model.debug, self.relation):
            # param = graph.add_parameter()
            param = Parameter(graph)
            param.abstract = model.abstract
            return param

    def call_sites(self, g):
        """Returns {call_site: eqv}.

        A call site C is either:

        * C = g(...)
        * eqv = {g}
        * C = switch(cond, g, g2)(...)
        * eqv = {g, g2}
        """
        call_sites = {cs: {g} for cs in g.call_sites}

        for node, key in g.higher_order_sites:
            if not (node.is_apply(P.switch) and (key == 2 or key == 3)):
                return None

            g2_node = node.inputs[5 - key]
            if not g2_node.is_constant_graph():
                return None

            g2 = g2_node.value
            uses = self.manager.uses[node]
            if not all(key2 == 0 for site, key2 in uses):
                return None

            call_sites.update({site: {g, g2} for site, _ in uses})

        return call_sites

    def _make_group(self, g, entries):
        valid = True
        ws = WorkSet([g])
        # Set of graphs that must all be augmented with the same fvs
        eqv = OrderedSet([g])
        # Call sites that will have to be updated
        call_sites = {}
        for currg in ws:
            eqv.add(currg)
            new_results = self.call_sites(currg)
            if new_results is None:
                valid = False
                break
            for _, new_eqv in new_results.items():
                ws.queue_all(new_eqv)
            call_sites.update(new_results)
        if valid:
            for gg in eqv:
                entries[gg] = NS(graph=gg, calls=call_sites, eqv=eqv)
                # Only the first graph gets the call sites
                call_sites = {}

    def make_groups(self):
        """Group the graphs according to their uses.

        Returns {graph: entry}.

        Each resulting entry contains the following fields:

        * graph: The graph for this entry.
        * eqv: A set of graphs that must be rewritten identically.
        * calls: A {call_site: graphs} dict that maps a node to the
            set of graphs that may be called there. That set may not
            include the graph for this entry. Only one graph in the
            eqv set will have a non-empty dictionary here.
        """
        entries = {}
        for g in self.graphs:
            if g in entries:
                continue
            self._make_group(g, entries)
        return entries

    def run(self):
        """Run the rewriting optimization."""
        # 1. Group the graphs and generate corresponding entries.
        entries = self.make_groups()

        # 2. Filter the entries to remove invalid ones. If an entry is invalid,
        # all entries corresponding to graphs in its eqv are also removed,
        # because it's all or nothing for an eqv group.
        if not getattr(self.filter, "noop", False):
            new_entries = dict(entries)
            for g, entry in entries.items():
                if g in new_entries and not self.filter(entry, entries):
                    for gg in entry.eqv:
                        del new_entries[gg]
            entries = new_entries

        # 3. Sort the entries, if operations must be done in a specific order
        # if self.order_key is NotImplemented:
        if getattr(self.order_key, "noop", False):
            tasks = entries.values()
        else:
            tasks = [entries[g] for g in sorted(entries, key=self.order_key)]

        chg = False

        # 4. Rewrite the call sites
        for entry in tasks:
            for node in entry.calls:
                chg |= self.rewrite_call(node, entry)

        # 5. Rewrite the graphs
        for entry in tasks:
            chg |= self.rewrite_graph(entry)

        return chg

    #######################
    # Methods to override #
    #######################

    @_noop
    def filter(self, entry, all_entries):
        """Return whether to rewrite this entry.

        It is allowed to add more data to the entry by setting additional
        fields into it.

        Arguments:
            entry: The entry to look at.
            all_entries: The complete dict of entries.
        """
        raise NotImplementedError("Override in subclass")

    @_noop
    def order_key(self, g):
        """Return a key to sort graphs.

        Graphs with a lower key will be processed first.

        Arguments:
            g: The graph to order.
        """
        raise NotImplementedError("Override in subclass")

    def rewrite_call(self, node, entry):
        """Rewrite the given call site.

        self.manager should be used to perform the rewriting, either using
        a transaction or directly.

        Arguments:
            node: A call site to rewrite.
            entry: An entry with the information needed to perform the rewrite.
                Note that entry.graph is not necessarily callable from this
                call site, but one or more of the graphs in entry.eqv are.

        Returns:
            True if any changes were made.
        """
        raise NotImplementedError("Override in subclass")

    def rewrite_graph(self, entry):
        """Rewrite the graph for this entry.

        The call sites are rewritten before the graphs.

        self.manager should be used to perform the rewriting, either using
        a transaction or directly. The parameters should be changed using
        the manager/transaction, not with `graph.add_parameter`.

        Arguments:
            entry: entry.graph is the graph to be rewritten.

        Returns:
            True if any changes were made.
        """
        raise NotImplementedError("Override in subclass")


class RemoveUnusedParameters(GraphInterfaceRewriter):
    def filter(self, entry, all_entries):
        """Keep the entry if graphs in eqv all miss common parameters."""
        params_grouped = zip(*[g.parameters for g in entry.eqv])
        entry.keep = [
            any(self.manager.uses[p] for p in params)
            for params in params_grouped
        ]
        # No rewrite if all parameters are kept
        return not all(entry.keep)

    def rewrite_call(self, call, entry):
        """Remove unused parameters from the call site."""
        new_call = call.graph.apply(
            call.inputs[0],
            *[arg for arg, keep in zip(call.inputs[1:], entry.keep) if keep]
        )
        new_call.abstract = call.abstract
        self.manager.replace(call, new_call)
        return True

    def rewrite_graph(self, entry):
        """Remove unused parameters from the graph parameters."""
        self.manager.set_parameters(
            entry.graph,
            [p for p, keep in zip(entry.graph.parameters, entry.keep) if keep],
        )
        return True


class LambdaLiftRewriter(GraphInterfaceRewriter):
    relation = "llift"

    def filter(self, entry, all_entries):
        g = entry.graph
        fvg = {
            g2
            for g2 in g.free_variables_total
            if isinstance(g2, Graph) and g2 not in all_entries
        }
        all_fvs = reduce(
            operator.or_,
            [gg.free_variables_extended for gg in entry.eqv],
            OrderedSet(),
        )
        if all_fvs and all(
            all(user in g.scope for user in g2.graph_users) for g2 in fvg
        ):
            entry.fvs = all_fvs
            entry.scope = {*g.scope, *fvg}
            return True
        else:
            return False

    @lru_cache(maxsize=None)
    def order_key(self, g):
        if g.parent:
            return self.order_key(g.parent) - 1
        else:
            return 0

    def rewrite_call(self, node, entry):
        fvs = [
            fv
            if any(fv in gg.free_variables_extended for gg in entry.calls[node])
            else make_dead(fv)
            for fv in entry.fvs
        ]
        new_node = node.graph.apply(*node.inputs, *fvs)
        new_node.abstract = node.abstract
        self.manager.replace(node, new_node)
        return True

    def rewrite_graph(self, entry):
        mng = self.manager
        new_params = list(entry.graph.parameters)
        with mng.transact() as tr:
            # Redirect the fvs to the parameter (those in scope)
            for fv in entry.fvs:
                param = self.param(entry.graph, fv)
                new_params.append(param)
                if fv in entry.graph.free_variables_extended:
                    for node, idx in mng.uses[fv]:
                        if node.graph in entry.scope:
                            tr.set_edge(node, idx, param)
            tr.set_parameters(entry.graph, new_params)
        return True


class GraphInterfaceRewriterOpt(Partializable):
    """Implements optimizer interface for GraphInferfaceRewriter."""

    def __init__(self, resources, rewriter):
        """Initialize GraphInterfaceRewriterOpt.

        Arguments:
            resources: The resources object associated to the pipeline.
            rewriter: A subclass of GraphInterfaceRewriter. It will be
                instantiated in the __call__ method.
        """
        self.resources = resources
        self.rewriter = rewriter
        self.name = rewriter.__name__

    def __call__(self, root):
        """Apply the rewriter on root."""
        mng = self.resources.opt_manager
        args = dict(opt=self, node=None, manager=mng, profile=False,)
        with tracer("opt", **args) as tr:
            tr.set_results(success=False, **args)
            mng.gc()
            rewriter = self.rewriter(mng)
            chg = rewriter.run()
            if chg:
                tracer().emit_success(**args, new_node=None)
            return chg


##########################
# LAMBDA LIFTING EXAMPLE #
##########################

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


__all__ = [
    "GraphInterfaceRewriter",
    "GraphInterfaceRewriterOpt",
    "RemoveUnusedParameters",
]
