"""Generate the gradient graph (augmented graph)."""


from collections import defaultdict
from functools import reduce
from typing import Set, Dict, Tuple, List

from .cconv import NestingAnalyzer
from .info import About
from .ir import Apply, Constant, Graph, ANFNode, \
    is_apply, is_constant, is_constant_graph
from .prim import ops as primops


add = Constant(primops.add)
J = Constant(primops.J)
index = Constant(primops.getitem)
cons = Constant(primops.cons_tuple)


def grad(graph):
    """Return the augmented graph. This is the same as the J primitive."""
    if graph.transforms.get('grad', None):
        return graph.transforms['grad']
    gr = Grad(graph)
    return gr.forward_graphs[graph]


class Grad:
    """Class to generate the gradient graph (augmented graph).

    Arguments:
        root: The root graph for which we want an augmented graph.
            We will operate on every graph in its scope, plus every
            graph called in its scope.
    """

    def __init__(self, root: Graph) -> None:
        """Initialize everything and process the gradient."""
        self.nest = NestingAnalyzer(root)
        assert not self.nest.parents()[root]

        self.fv_order: Dict[Graph, List[ANFNode]] = defaultdict(list)
        for g, fvs in self.nest.free_variables_total().items():
            self.fv_order[g] = list(fvs)

        # g -> ▶g
        self.forward_graphs: Dict[Graph, Graph] = {}
        # g -> ◀g
        self.backpropagator_graphs: Dict[Graph, Graph] = {}

        # x -> ▶x
        self.forward_nodes: Dict[ANFNode, ANFNode] = {}
        # x -> ◀x
        self.backpropagator_nodes: Dict[ANFNode, ANFNode] = {}
        # (x, g) -> ∇x (in the context of that graph)
        self.sensitivity_nodes: Dict[Tuple[ANFNode, Graph], ANFNode] = {}
        # x -> ◀x(∇x) (for x an Apply node)
        self.step_nodes: Dict[Apply, Apply] = {}

        # To get the uses of a graph, we need to know which Constant(s)
        # refer to that graph, so we keep that in this map.
        self.graph_to_ct: Dict[Graph, Set[Constant]] = defaultdict(set)

        graphs = self.nest.scopes()[root]

        for g in graphs:
            self._make_forward_graph(g)
            self._make_backpropagator_graph(g)

        for g in graphs:
            self._process_graph_forward(g)

        for g in graphs:
            self._process_graph_backward(g)

    def _make_forward_graph(self, graph: Graph) -> None:
        # Forward graph
        with About(graph.debug, 'grad_fprop'):
            fgraph = Graph()
        graph.transforms['grad'] = fgraph
        fgraph.transforms['primal'] = graph
        self.forward_graphs[graph] = fgraph
        # Same parameters as the original, but tagged as forward
        for p in graph.parameters:
            with About(p.debug, 'grad_fprop'):
                self.forward_nodes[p] = fgraph.add_parameter()

    def _make_backpropagator_graph(self, graph: Graph) -> None:
        # Backpropagator graph
        with About(graph.debug, 'grad_bprop'):
            bgraph = Graph()
        self.backpropagator_graphs[graph] = bgraph
        # Takes output sensitivity as sole parameter
        with About(graph.debug, 'grad_sens'):
            bparam = bgraph.add_parameter()
            self.sensitivity_nodes[(graph.output, graph)] = bparam

    def _make_cons(self, graph, *elems):
        if len(elems) == 0:
            return Constant(())
        else:
            x, *rest = elems
            return graph.apply(primops.cons_tuple,
                               x,
                               self._make_cons(graph, *rest))

    def _apply(self, graph, *inputs):
        wrapped_inputs = [i if isinstance(i, ANFNode) else Constant(i)
                          for i in inputs]
        return Apply(wrapped_inputs, graph)

    def _process_graph_forward(self, graph):
        """Create the forward graph."""
        fgraph = self.forward_graphs[graph]
        bgraph = self.backpropagator_graphs[graph]

        # Return (▶graph.output, ◀graph). The first element is given
        # by the `phi` method.

        fgraph.output = self._make_cons(
            fgraph,
            self.phi(graph.output),
            bgraph
        )

    def phi(self, node):
        """Compute equivalent node in forward graph."""
        if node in self.forward_nodes:
            return self.forward_nodes[node]

        fg = node.graph and self.forward_graphs[node.graph]

        if is_constant_graph(node) and node.value in self.forward_graphs:
            # We will have to process this graph too.
            fwd = self.forward_graphs[node.value]
            bprop = self.backpropagator_graphs[node.value]
            self.graph_to_ct[node.value].add(node)
        elif is_constant(node):
            # Note that this application will have its graph set to None, which
            # makes sense since it's basically a constant expression.
            fwd, bprop = self._apply(fg, J, node), None
        elif is_apply(node) and node.graph not in self.forward_graphs:
            fwd, bprop = self._apply(fg, J, node), None
        elif is_apply(node):
            # a = f(x, y) -> ▶a, ◀a = ▶f(▶x, ▶y)
            fwd_args = [self.phi(n) for n in node.inputs]
            app = self._apply(fg, *fwd_args)
            # ▶a (the first element)
            fwd = self._apply(fg, index, app, 0)
            # ◀a (the second element)
            # Note that ◀a is not part of the forward graph, however,
            # it will be a free variable of the backpropagator graph.
            bprop = self._apply(fg, index, app, 1)
        else:
            # Note: Parameters were all added to forward_nodes in
            # scaffold_graph, so they won't trigger this branch.
            raise Exception('This should be unreachable.')  # pragma: no cover

        self.forward_nodes[node] = fwd
        self.backpropagator_nodes[node] = bprop
        if not fwd.debug.about:
            fwd.debug.about = About(node.debug, 'grad_fprop')
        if bprop and not bprop.debug.about:
            bprop.debug.about = About(node.debug, 'grad_bprop')
        return fwd

    def _process_graph_backward(self, graph):
        """Create the backward graph."""
        bgraph = self.backpropagator_graphs[graph]

        # Return ((∇fv1, ∇fv2, ...), ∇arg1, ∇arg2, ...)
        # Where ∇x is given by `rho(x, graph)`

        bgraph.output = self._make_cons(
            bgraph,
            self._make_cons(bgraph,
                            *[self.rho(p, graph)
                              for p in self.fv_order[graph]]),
            *[self.rho(p, graph) for p in graph.parameters]
        )

    def bprop_step(self, node):
        """Compute backpropagator expression for this node.

        If node is a = f(x, y), this returns ◀a(∇a). That expression returns
        gradient contributions to ∇f, ∇x and ∇y.
        """
        if node in self.step_nodes:
            return self.step_nodes[node]

        bg = node.graph and self.backpropagator_graphs[node.graph]
        bprop = self.backpropagator_nodes[node]
        assert bprop
        rval = self._apply(bg, bprop, self.rho(node, node.graph))
        self.step_nodes[node] = rval
        if not rval.debug.about:
            rval.debug.about = About(node.debug, 'grad_bprop_step') \
                # pragma: no cover
        return rval

    def rho(self, node, graph):
        """Compute expression for gradient wrt node and graph."""
        # We index with node and graph because the same node may have multiple
        # sensitivity variables, one for each graph that refers to the
        # original.
        key = (node, graph)
        if key in self.sensitivity_nodes:
            return self.sensitivity_nodes[key]

        bg = self.backpropagator_graphs[graph]

        # We will accumulate all gradient contributions here.
        contribs = []

        for user, idx in node.uses:
            # Each use of a variable contributes to its gradient.
            if user.graph is graph:
                # A use in the same graph: we get the backpropagator expression
                # and we use the argument index to extract the right
                # contribution.
                with About(node.debug, 'grad_sens'):
                    contrib = self._apply(bg,
                                          index,
                                          self.bprop_step(user), idx)
                contribs.append(contrib)
            else:
                # Uses in different graphs are ignored here. We take them
                # into account below.
                pass  # pragma: no cover

        # We list all graphs that are immediately nested in this one and have
        # this node as a free variable. These graphs may not technically
        # contain a use of node, but they contain graphs that do, so there
        # is a gradient of the node with respect to them.
        children = {g for g in self.nest.coverage()
                    if self.nest.parents()[g] is graph
                    and node in self.nest.free_variables_total()[g]}

        for child in children:
            # This is the index of this node in the graph's free
            # variables.
            idx = self.fv_order[child].index(node)
            for graph_ct in self.graph_to_ct[child]:
                # We get the sensitivity wrt the closure using `rho` on the
                # constant and this graph. Concretely this means we will
                # look at the uses of the closure in this graph. Or if the
                # closure is returned, this will be the sensitivity wrt the
                # output. We index this sensitivity with idx to get the
                # contribution we seek.
                contrib = self._apply(bg,
                                      index,
                                      self.rho(graph_ct, graph),
                                      idx)
                contribs.append(contrib)

        # NOTE: The order of nodes in contribs is not deterministic, because
        # the order of users isn't. In theory that doesn't matter, because we
        # add them all, but in practice there could be slight numerical
        # differences.

        if len(contribs) == 0:
            # No contributions means a gradient of zero, naturally.
            sens = self._apply(bg, primops.zeros_like,
                               self.forward_nodes[node])
        else:
            # Contributions must be added together.
            def mkadd(x, y):
                return self._apply(bg, add, x, y)
            sens = reduce(mkadd, contribs)

        if not sens.debug.about:
            sens.debug.about = About(node.debug, 'grad_sens')
        self.sensitivity_nodes[node] = sens
        return sens
