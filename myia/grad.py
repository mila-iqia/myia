"""Generate the gradient graphs"""


from collections import defaultdict
from functools import reduce
from typing import Set, Dict, Tuple, List

from .info import About
from .ir import Apply, Constant, Graph, ANFNode, \
    is_apply, is_constant, is_constant_graph, is_parameter, \
    manage, clone
from .opt import sexp_to_node
from .prim import ops as primops
from .utils import Partializable


add = Constant(primops.add)
J = Constant(primops.J)
index = Constant(primops.getitem)
cons = Constant(primops.cons_tuple)


def _make_cons(graph, *elems):
    if len(elems) == 0:
        return Constant(())
    else:
        x, *rest = elems
        return graph.apply(primops.cons_tuple,
                           x,
                           _make_cons(graph, *rest))


class GraphRemapper(Partializable):

    def __init__(self,
                 relation,
                 graphs,
                 remappers=None,
                 master=None,
                 in_remapper=None):
        self.relation = relation
        self.graphs = graphs
        self.graph_repl = {}
        self.repl = {}
        self.remappers = remappers
        self.in_remapper = in_remapper or self
        self.master = master or self
        self.to_link = []
        self._populated = False

    def prepare(self):
        if isinstance(self.in_remapper, str):
            self.in_remapper = self.remappers[self.in_remapper]
        if isinstance(self.master, str):
            self.master = self.remappers[self.master]

    def get(self, g, node):
        if (g, node) in self.repl:
            return self.repl[(g, node)]
        elif node in self.repl:
            return self.repl[node]
        elif is_constant_graph(node):
            return self.repl[node.value]
        else:
            raise KeyError(f'Unprocessed node: {node}')

    def get_graph(self, g):
        return self.graph_repl[g]

    def add_node(self, key, g, node, ng, new_node, link=None):
        if link is None:
            link = is_apply(new_node)
        self.repl[key] = new_node
        if link:
            self.to_link.append((key, g, node, ng, new_node))

    def gen_graph(self, g):
        if self.master is not self:
            ng = self.master.get_graph(g)
        else:
            with About(g.debug, self.relation):
                ng = Graph()
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        if self.master is not self:
            return
        with About(p.debug, self.relation):
            # self.repl[p] = ng.add_parameter()
            self.add_node(p, g, p, ng, ng.add_parameter())

    def gen_apply(self, g, ng, node):
        with About(node.debug, self.relation):
            # self.repl[node] = ng.apply()
            self.add_node(node, g, node, ng, ng.apply())

    def gen_child(self, g, ng, child):
        pass

    def gen_fv(self, g, ng, node):
        pass

    def gen_fv_graph(self, g, ng, g2):
        pass

    def gen_constant(self, g, ng, ct):
        if self.master is not self:
            return
        with About(ct.debug, self.relation):
            self.add_node(ct, g, ct, ng, ct)

    def gen_constant_graph(self, g, ng, ct):
        return self.gen_constant(g, ng, ct)

    def link_apply(self, g, ng, node, new_node):
        new_inputs = [self.in_remapper.get(g, inp)
                      for inp in node.inputs]
        new_node.inputs = new_inputs

    def finalize_graph(self, g, ng):
        ng.output = self.get(g, g.output)

    def populate(self):
        if self._populated:
            return

        for g in self.graphs:
            ng = self.gen_graph(g)

        for g in self.graphs:
            ng = self.get_graph(g)
            for p in g.parameters:
                self.gen_parameter(g, ng, p)

            for node in g.nodes:
                if is_apply(node):
                    self.gen_apply(g, ng, node)

            for child in g.children:
                self.gen_child(g, ng, child)

            for node in g.free_variables_indirect:
                self.gen_fv(g, ng, node)

            for graph in g.free_variables_graphs:
                self.gen_fv_graph(g, ng, graph)

            for ct in g.constants:
                if is_constant_graph(ct):
                    self.gen_constant_graph(g, ng, ct)
                else:
                    self.gen_constant(g, ng, ct)

        self._populated = True

    def link(self):
        for _, g, node, ng, new_node in self.to_link:
            self.link_apply(g, ng, node, new_node)

    def finalize(self):
        if self.master is self:
            for g in self.graphs:
                self.finalize_graph(g, self.get_graph(g))


class FPropAppRemapper(GraphRemapper):
    pass


class FPropRemapper(GraphRemapper):
    def gen_constant(self, g, ng, ct):
        self.repl[(g, ct)] = sexp_to_node((primops.J, ct), ng)

    def gen_constant_graph(self, g, ng, ct):
        if ct.value in self.graphs:
            new_ct = Constant(self.get_graph(ct.value))
            self.repl[ct] = new_ct
            self.repl[ct.value] = new_ct
        else:
            self.gen_constant(g, ng, ct)

    def gen_fv(self, g, ng, fv):
        if fv.graph not in self.graphs:
            return self.gen_constant(g, ng, fv)

    def gen_fv_graph(self, g, ng, fvg):
        if fvg in self.graphs:
            return self.gen_constant_graph(g, ng, Constant(fvg))
        else:
            return self.gen_constant(g, ng, fvg)

    def link_apply(self, g, ng, node, new_node):
        assert not is_parameter(node)
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node((primops.getitem, app, 0), ng).inputs

    def finalize_graph(self, g, ng):
        g.transforms['grad'] = ng
        ng.transforms['primal'] = g
        elems = self.get(g, g.output), self.remappers['grad_sens'].get_graph(g)
        ng.output = _make_cons(ng, *elems)


class BPropRemapper(GraphRemapper):
    def link_apply(self, g, ng, node, new_node):
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node((primops.getitem, app, 1), ng).inputs


class BPropAppRemapper(GraphRemapper):
    def link_apply(self, g, ng, node, new_node):
        if is_parameter(node):
            return
        fn = self.remappers['grad_bprop'].get(g, node)
        arg = self.remappers['grad_sens'].get(g, node)
        new_node.inputs = [fn, arg]


class SensRemapper(GraphRemapper):

    def __init__(self, relation, graphs, remappers=None):
        super().__init__(relation, graphs, remappers)
        self.fv_order = {g: list(g.free_variables_total) for g in graphs}

    def gen_graph(self, g):
        with About(g.debug, 'grad_bprop'):
            ng = Graph()
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        self.gen_apply(g, ng, p)

    def gen_apply(self, g, ng, node):
        with About(node.debug, self.relation):
            if node is g.output:
                new_node = ng.add_parameter()
            else:
                new_node = ng.apply()
        self.add_node((g, node), g, node, ng, new_node)

    def gen_child(self, g, ng, child):
        with About(child.debug, self.relation):
            self.add_node((g, child), g, child, ng, ng.apply())

    def gen_fv(self, g, ng, node):
        with About(node.debug, self.relation):
            self.add_node((g, node), g, node, ng, ng.apply())

    def gen_fv_graph(self, g, ng, g2):
        with About(g2.debug, self.relation):
            self.add_node((g, g2), g, g2, ng, ng.apply())

    def link_apply(self, g, ng, node, new_node):
        mng = g.manager
        assert not is_parameter(new_node)

        contribs = []

        if isinstance(node, Graph):
            uses = set()
            for ct in g.constants:
                if ct.value is node:
                    uses |= mng.uses[ct]
        else:
            uses = mng.uses[node]

        for user, key in uses:
            if user.graph is g:
                if user is user.graph.return_:
                    if len(ng.parameters) == 0:
                        with About(g.output.debug, 'grad_sens'):
                            ng.add_parameter()
                    sexp = (primops.identity, ng.parameters[0])
                    contribs.append(sexp)
                    continue
                src = self.remappers['grad_bprop_app'].get(g, user)
                sexp = (primops.getitem, src, key)
                contribs.append(sexp)

        # TODO: deconstruct nested graphs

        children = {g2 for g2 in self.graphs
                    if g2.parent is g
                    and node in g2.free_variables_total}

        for child in children:
            idx = self.fv_order[child].index(node)
            assert (g, child) in self.repl
            sexp = (primops.getitem, self.get(g, child), idx)
            contribs.append(sexp)

        n = len(contribs)
        if n == 0:
            sexp = (primops.zeros_like,
                    (primops.Jinv,
                     self.remappers['grad_fprop'].get(g, node)))
        else:
            def mkadd(x, y):
                return (primops.add, x, y)
            sexp = reduce(mkadd, contribs)

        new_node.inputs = sexp_to_node(sexp, ng).inputs

    def finalize_graph(self, g, ng):
        fv_sens = [self.get(g, fv) for fv in g.free_variables_total]
        in_sens = [self.get(g, p) for p in g.parameters]
        ng.output = _make_cons(ng, _make_cons(ng, *fv_sens), *in_sens)
        if len(ng.parameters) == 0:
            with About(g.output.debug, 'grad_sens'):
                ng.add_parameter()


class RemapperSet:
    def __init__(self, graphs, **remappers):
        self.remappers = {
            name: remapper(relation=name, graphs=graphs, remappers=self)
            for name, remapper in remappers.items()
        }
        self.graphs = graphs

    def run(self):
        for _, remapper in self.remappers.items():
            remapper.prepare()
        for _, remapper in self.remappers.items():
            remapper.populate()
        for _, remapper in self.remappers.items():
            remapper.link()
        for _, remapper in self.remappers.items():
            remapper.finalize()

    def __getitem__(self, item):
        return self.remappers[item]


def grad(graph, mng=None):
    if mng is None:
        mng = manage(graph)
    # if graph.transforms.get('grad', None):
    #     return graph.transforms['grad']
    mng.add_graph(graph)
    res = Grad(mng, graph).result
    return clone(res)


class Grad:
    def __init__(self, mng, root):
        graphs = root.scope

        remappers = RemapperSet(
            graphs,
            grad_fprop=FPropRemapper.partial(),
            grad_fprop_app=FPropAppRemapper.partial(
                master='grad_fprop',
                in_remapper='grad_fprop'
            ),
            grad_bprop=BPropRemapper.partial(
                master='grad_fprop'
            ),
            grad_sens=SensRemapper.partial(
            ),
            grad_bprop_app=BPropAppRemapper.partial(
                master='grad_sens'
            ),
        )
        remappers.run()
        self.result = remappers['grad_fprop'].get_graph(root)


def old_grad(graph):
    """Return the forward graph, which embeds its backprop in a closure.

    This is the same as the J primitive, when the argument is a graph."""
    if graph.transforms.get('grad', None):
        return graph.transforms['grad']
    gr = Grad(graph)
    g = gr.forward_graphs[graph]

    from .graph_utils import dfs
    from .ir import succ_deeper
    for node in list(dfs(g.output, succ_deeper)):
        if is_apply(node):
            for i, inp in enumerate(node.inputs):
                if is_apply(inp) and inp.graph is None:
                    node.inputs[i] = node.graph.apply(*inp.inputs)

    return g


class OldGrad:
    """Class to generate the gradient graph.

    This returns the forward graph, which returns a pair of the original value
    coupled with a closure that performs backpropagation.

    Arguments:
        root: The root graph for which we want the gradient graph.
            We will operate on every graph in its scope, plus every
            graph called in its scope.
    """

    def __init__(self, root: Graph) -> None:
        """Initialize everything and process the gradient."""
        self.mng = manage(root)
        assert not self.mng.parents[root]

        self.fv_order: Dict[Graph, List[ANFNode]] = defaultdict(list)
        for g, fvs in self.mng.free_variables_total.items():
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

        graphs = self.mng.scopes[root]

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

        for user, idx in self.mng.uses[node]:
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
        children = {g for g in self.mng.graphs
                    if self.mng.parents[g] is graph
                    and node in self.mng.free_variables_total[g]}

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
