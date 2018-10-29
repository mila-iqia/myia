"""Generate the gradient graphs for reverse mode.

The J transform on a graph produces two graphs, the forward graph (fprop)
and the backpropagator graph (bprop). The former returns the result of the
computation and the bprop graph, and the latter takes output sensitivities
and returns input sensitivities. For each node `y = f(x)` of the graph,
we generate:

* Three nodes in the fprop graph, using one GraphRemapper and two
  SlaveRemappers (it doesn't really matter which one is the master and
  which ones are the slaves):

    temp = fprop_f(fprop_x)  # FPropAppRemapper (slave)
    fprop_y = temp[0]        # FPropRemapper    (master)
    bprop_y = temp[1]        # BPropRemapper    (slave)

* Two nodes in the bprop graph, using one GraphRemapper and one
  SlaveRemapper:

    # BPropAppRemapper (slave)
    from_y = bprop_y(sens_y)

    # SensRemapper (master)
    sens_y = hyper_add(from_z[idx] for z, idx in uses(y))

* For the output node, in the fprop graph, we generate:

    return fprop_y, bprop_y

* The output of the bprop graph contains the sensitivities of
  the free variables and parameters. Sensitivities for free variables
  are stored in an Env:

    sens_allfvs = env_setitem(newenv,      embed(fv1), sens_fv1)
    sens_allfvs = env_setitem(sens_allfvs, embed(fv2), sens_fv2)
    ...
    return sens_allfvs, sens_param1, sens_param_2, ...

All remappers generate their nodes first, and then the nodes are linked
with each other, which allows them to refer to each other.

SensRemapper is the most complex one, mainly because it walks through
each node's uses and has to deal with free variables. For any given node,
SensRemapper will generate a sensitivity node in each graph that uses
it, so e.g. in `lambda x: x + (lambda y: x + y)(123)` two different
sensitivity nodes are created for x, because it is used in both lambda
expressions. See `SensRemapper.link_apply` for more information.
"""


from functools import reduce

from .composite import zeros_like, hyper_add
from .info import About
from .ir import Constant, Graph, clone
from .opt import sexp_to_node
from .prim import ops as primops, Primitive
from .prim.grad_implementations import augmented_graphs
from .utils import Partializable, overload, newenv


class GraphRemapper(Partializable):
    """Maps every node of a graph to a new node in a different graph.

    Remapping rules can be adapted in subclasses.

    Arguments:
        relation: The relation between the original node and the new node.
        graphs: The graphs to transform.
        remappers (Optional): A RemapperSet for if the remapped nodes need
            to refer to nodes in other remappers.
        graph_relation (Optional): The relation between the original graph
            and the new graph (defaults to relation).
    """

    def __init__(self,
                 relation,
                 graphs,
                 *,
                 remappers=None,
                 graph_relation=None):
        """Initialize a GraphRemapper."""
        self.relation = relation
        self.graph_relation = graph_relation or relation
        self.graphs = graphs
        self.graph_repl = {}
        self.repl = {}
        self.remappers = remappers
        self.to_link = []

    def get(self, g, node):
        """Get the new node corresponding to the given (graph, node) pair.

        The (g, node) pair corresponds to a use of a node from graph g.
        The node may or may not belong to g. Some remappers may ignore
        g.
        """
        if (g, node) in self.repl:
            return self.repl[(g, node)]
        elif node in self.repl:
            return self.repl[node]
        else:
            raise AssertionError(f'Unprocessed node: {node}')

    def get_graph(self, g):
        """Get the new graph corresponding to g."""
        return self.graph_repl[g]

    def add_node(self, key, g, node, ng, new_node, link=None):
        """Remap a node.

        Arguments:
            key: Either (g, node) or just a node. The latter case corresponds
                to remapping all uses of node from all graphs to the same
                node.
            g: The graph to which node belongs.
            node: The node to remap.
            ng: Equivalent to self.get_graph(g).
            new_node: What to remap node to.
            link: Whether to link that node or not using link_apply. By default
                it is True if the node is an Apply node.
        """
        if link is None:
            link = new_node.is_apply()
        self.repl[key] = new_node
        if link:
            self.to_link.append((key, g, node, ng, new_node))

    def gen_graph(self, g):
        """Generate a new graph corresponding to g."""
        with About(g.debug, self.graph_relation):
            ng = Graph()
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        """Generate the node for parameter p of g."""
        with About(p.debug, self.relation):
            self.add_node(p, g, p, ng, ng.add_parameter())

    def gen_apply(self, g, ng, node):
        """Generate the node for application node of g."""
        with About(node.debug, self.relation):
            self.add_node(node, g, node, ng, ng.apply())

    def gen_child(self, g, ng, child):
        """Generate the node for a child graph of g."""
        pass

    def gen_fv(self, g, ng, node):
        """Generate the node for free variable node of g."""
        pass

    def gen_fv_graph(self, g, ng, g2):
        """Generate the node for free variable graph g2 of g."""
        pass

    def gen_constant(self, g, ng, ct):
        """Generate the node for constant ct used by g."""
        with About(ct.debug, self.relation):
            self.add_node(ct, g, ct, ng, ct)

    def gen_constant_graph(self, g, ng, ct):
        """Generate the node for constant graph ct used by g."""
        return self.gen_constant(g, ng, ct)

    def link_apply(self, g, ng, node, new_node):
        """Generate the inputs for new_node."""
        raise NotImplementedError()

    def finalize_graph(self, g, ng):
        """Finalize the new graph ng (set its output)."""
        raise NotImplementedError()

    def populate(self):
        """Generate all necessary nodes."""
        for g in self.graphs:
            ng = self.gen_graph(g)

        for g in self.graphs:
            ng = self.get_graph(g)
            for p in g.parameters:
                self.gen_parameter(g, ng, p)

            for node in g.nodes:
                if node.is_apply():
                    self.gen_apply(g, ng, node)

            for child in g.children:
                self.gen_child(g, ng, child)

            for node in g.free_variables_nodes:
                self.gen_fv(g, ng, node)

            for graph in g.free_variables_graphs:
                self.gen_fv_graph(g, ng, graph)

            for ct in g.constants:
                if ct.is_constant_graph():
                    self.gen_constant_graph(g, ng, ct)
                else:
                    self.gen_constant(g, ng, ct)

    def link(self):
        """Link all nodes to their new inputs."""
        for _, g, node, ng, new_node in self.to_link:
            self.link_apply(g, ng, node, new_node)

    def finalize(self):
        """Finalize the graphs that belong to the remapper."""
        for g in self.graphs:
            self.finalize_graph(g, self.get_graph(g))


class SlaveRemapper(GraphRemapper):
    """Maps every node of a graph to new nodes in the master's graphs.

    Arguments:
        relation: The relation between the original node and the new node.
        graphs: The graphs to transform.
        master: The name of a remapper whose graphs this remapper
            will generate nodes into.
        remappers (Optional): A RemapperSet for if the remapped nodes need
            to refer to nodes in other remappers.
        graph_relation (Optional): The relation between the original graph
            and the new graph (defaults to relation).
    """

    def __init__(self,
                 relation,
                 graphs,
                 master,
                 *,
                 remappers=None,
                 graph_relation=None):
        """Initialize a SlaveRemapper."""
        super().__init__(
            relation, graphs, remappers=remappers,
            graph_relation=graph_relation
        )
        self._master_name = master

    @property
    def master(self):
        """Remapper providing the graphs to generate into."""
        return self.remappers[self._master_name]

    def gen_graph(self, g):
        """Get the master's graph for g."""
        ng = self.master.get_graph(g)
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        """Parameters are generated by the master."""
        pass

    def gen_constant(self, g, ng, ct):
        """Constants are generated by the master."""
        pass

    def finalize(self):
        """Only the master finalizes."""
        pass


class FPropAppRemapper(SlaveRemapper):
    """Generate applications in the forward graph.

    This is transform A, generating into transform B's graph.

    x = a(b, c) => A:x = (B:a)(B:b, B:c)
    """

    def link_apply(self, g, ng, node, new_node):
        """Link generated nodes to their inputs.

        x = a(b, c) => A:x = (B:a)(B:b, B:c)
        """
        new_inputs = [self.remappers['grad_fprop'].get(g, inp)
                      for inp in node.inputs]
        new_node.inputs = new_inputs


class FPropRemapper(GraphRemapper):
    """Generate nodes in the forward graph.

    This is transform B.

    x = a(b, c) => B:x = (A:x)[0]
    """

    def gen_constant(self, g, ng, ct):
        """Constants are wrapped with a call to J."""
        self.repl[(g, ct)] = sexp_to_node((primops.J, ct), ng)

    def gen_constant_graph(self, g, ng, ct):
        """Constant graphs map to their remapped versions.

        Graphs that are not remapped are wrapped with J.
        """
        if ct.value in self.graphs:
            new_ct = Constant(self.get_graph(ct.value))
            self.repl[ct] = new_ct
            self.repl[ct.value] = new_ct
        else:
            self.gen_constant(g, ng, ct)

    def gen_fv(self, g, ng, fv):
        """Free variables outside the remapped scope are wrapped with J.

        Remapped free variables are remapped elsewhere.
        """
        if fv.graph not in self.graphs:
            return self.gen_constant(g, ng, fv)

    def gen_fv_graph(self, g, ng, fvg):
        """Free variables that are graphs are handled like constants."""
        if fvg in self.graphs:
            return self.gen_constant_graph(g, ng, Constant(fvg))
        else:
            return self.gen_constant(g, ng, fvg)

    def link_apply(self, g, ng, node, new_node):
        """Link generated nodes to their inputs.

        x = a(b, c) => B:x = (A:x)[0]
        """
        assert not node.is_parameter()
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node(
            (primops.tuple_getitem, app, 0),
            ng
        ).inputs

    def finalize_graph(self, g, ng):
        """We generate the pair (B:output, E:g)."""
        g.transforms['grad'] = ng
        ng.transforms['primal'] = g
        out = self.get(g, g.output)
        bprop = self.remappers['grad_sens'].get_graph(g)
        ng.output = ng.apply(primops.make_tuple, out, bprop)

    def get_jinv(self, node):
        """Generate Jinv(B:node)."""
        if (node, 'jinv') not in self.repl:
            if isinstance(node, Graph):
                if node not in self.graphs:
                    new_node = Constant(node)
                else:
                    assert node.parent is not None
                    ng = self.get_graph(node.parent)
                    ct = Constant(self.get_graph(node))
                    with About(node.debug, 'equiv'):
                        new_node = ng.apply(primops.Jinv, ct)
            else:
                if node.graph not in self.graphs:
                    new_node = node
                else:
                    ng = self.get_graph(node.graph)
                    node2 = self.get(None, node)
                    with About(node.debug, 'equiv'):
                        new_node = ng.apply(primops.Jinv, node2)
            self.repl[node, 'jinv'] = new_node
        return self.repl[node, 'jinv']


class BPropRemapper(SlaveRemapper):
    """Generate backpropagators in the forward graph.

    This is transform C.

    x = a(b, c) => C:x = (A:x)[1]
    """

    def link_apply(self, g, ng, node, new_node):
        """Link generated nodes to their inputs.

        x = a(b, c) => C:x = (A:x)[1]
        """
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node(
            (primops.tuple_getitem, app, 1),
            ng
        ).inputs


class BPropAppRemapper(SlaveRemapper):
    """Generate the reverse applications in the backward graph.

    This is transform D, generating into transform E's graph.

    x = a(b, c) => D:x = (C:x)(E:x)
    """

    def link_apply(self, g, ng, node, new_node):
        """Link generated nodes to their inputs.

        x = a(b, c) => D:x = (C:x)(E:x)
        """
        assert not node.is_parameter()
        fn = self.remappers['grad_bprop'].get(g, node)
        arg = self.remappers['grad_sens'].get(g, node)
        new_node.inputs = [fn, arg]


class SensRemapper(GraphRemapper):
    """Generate the sensitivities in the backward graph.

    This is transform E.

    x, used by y at index i and z at index j =>
        E:x = D:y[i] + D:z[j]
    """

    def gen_parameter(self, g, ng, p):
        """Generate nodes for parameter sensitivities.

        This graph is reversed, so parameter sensitivities are outputs,
        not parameters of ng.
        """
        self.gen_apply(g, ng, p)

    def gen_apply(self, g, ng, node):
        """Generate sensitivities for applications.

        * The output node's sensitivity is ng's sole parameter.
        * If a node is used in multiple graphs, each graph has a
          corresponding sensitivity node.
        """
        with About(node.debug, self.relation):
            if node is g.output:
                new_node = ng.add_parameter()
            else:
                new_node = ng.apply()
        # NOTE: First parameter to add_node is (g, node) instead of just node.
        # This lets us dispatch to a different node depending on whether it
        # belongs to the graph that uses it, or is a free variable.
        self.add_node((g, node), g, node, ng, new_node)

    def gen_child(self, g, ng, child):
        """Generate sensitivities for child graphs."""
        with About(child.debug, self.relation):
            self.add_node((g, child), g, child, ng, ng.apply())

    def gen_fv(self, g, ng, node):
        """Generate sensitivities for free variables.

        Note that the default gen_fv does nothing, so this is different
        behavior.
        """
        with About(node.debug, self.relation):
            self.add_node((g, node), g, node, ng, ng.apply())

    def gen_fv_graph(self, g, ng, g2):
        """Generate sensitivities for free variables that are graphs."""
        with About(g2.debug, self.relation):
            self.add_node((g, g2), g, g2, ng, ng.apply())

    def link_apply(self, g, ng, node, new_node):
        """Link generated nodes to their inputs.

        x, used by y at index i and z at index j =>
            E:x = D:y[i] + D:z[j]
        """
        mng = g.manager
        assert not new_node.is_parameter()

        if isinstance(node, Graph):
            # This was added via gen_child or gen_fv_graph
            uses = set()
            for ct in g.constants:
                if ct.value is node:
                    uses |= mng.uses[ct]
        else:
            uses = mng.uses[node]

        contribs = []

        for user, key in uses:
            if user.graph is g:
                # We only concern ourselves with uses in this graph
                if user is user.graph.return_:
                    # This is the graph's output, so the contribution
                    # is the output sensitivity, which is contained in
                    # ng's sole parameter.
                    if len(ng.parameters) == 0:
                        # This will happen if the graph returns a free
                        # variable directly.
                        with About(g.output.debug, 'grad_sens'):
                            ng.add_parameter()
                    # We need to call identity because we need to modify
                    # new_node's inputs at the end of the function, we can't
                    # simply replace it.
                    sexp = (primops.identity, ng.parameters[0])
                    contribs.append(sexp)
                else:
                    # If the application is e.g. z = f(x, y), BPropAppRemapper
                    # calculates the tuple (df, dx, dy) = backpropagator_f(dz)
                    # If we are processing node f, x or y, we will respectively
                    # get element 0, 1 or 2 of that tuple and add that to our
                    # contribs list.
                    src = self.remappers['grad_bprop_app'].get(g, user)
                    sexp = (primops.tuple_getitem, src, key)
                    contribs.append(sexp)

        # This is equivalent to the original node.Note that we aren't really
        # interested in the node's value: jinv is used along with embed and
        # zeros_like, which only care about the original node's inferred type
        # and shape.
        jinv = self.get_jinv(node)

        # TODO: deconstruct nested graphs
        # TODO: figure out what I meant by "deconstruct nested graphs" :(

        # These are all the graphs nested in g which have this node as a
        # free variable. Each of these graphs has a sensitivity node, and
        # we will extract contributions from them.
        children = {g2 for g2 in self.graphs
                    if g2.parent is g
                    and node in g2.free_variables_total}

        for child in children:
            assert (g, child) in self.repl
            sexp = (primops.env_getitem,
                    self.get(g, child),
                    # This represents the node's "key" into the env.
                    (primops.embed, jinv),
                    # This is the default, if there is no entry for this key.
                    (zeros_like, jinv))
            contribs.append(sexp)

        n = len(contribs)
        if n == 0:
            sexp = (zeros_like, jinv)
        else:
            # All contributions are added together with hyper_add.
            def mkadd(x, y):
                return (hyper_add, x, y)
            sexp = reduce(mkadd, contribs)

        new_node.inputs = sexp_to_node(sexp, ng).inputs

    def get_jinv(self, node):
        """Generate Jinv(B:node) (shortcut).

        This is essentially equivalent to the original node. We can't use the
        original node directly because the graph it belongs to is not available
        any more after the transform.
        """
        return self.remappers['grad_fprop'].get_jinv(node)

    def finalize_graph(self, g, ng):
        """Generate the output of the backprop graph.

        * Sensitivities for all free variables are packed in an
          EnvInstance using env_setitem.
        * We return a tuple with fv sensitivities first, and then
          all parameter sensitivities.
        """
        fv_sens = Constant(newenv)
        for fv in g.free_variables_total:
            sens = self.get(g, fv)
            # NOTE: If sens is an application of zeros_like, it would be
            # possible to skip adding it to the env.
            fv_sens = ng.apply(
                primops.env_setitem,
                fv_sens,
                ng.apply(primops.embed, self.get_jinv(fv)),
                sens
            )
        in_sens = [self.get(g, p) for p in g.parameters]
        ng.output = ng.apply(primops.make_tuple,
                             fv_sens,
                             *in_sens)
        if len(ng.parameters) == 0:
            # This can happen if the output is a constant. In that case we just
            # add a dummy parameter to satisfy the backpropagator protocol.
            with About(g.output.debug, 'grad_sens'):
                ng.add_parameter()


class RemapperSet:
    """Set of remappers working together to generate one or more graphs."""

    def __init__(self, graphs, **remappers):
        """Initialize a RemapperSet."""
        self.remappers = {
            name: remapper(relation=name, graphs=graphs, remappers=self)
            for name, remapper in remappers.items()
        }
        self.graphs = graphs

    def run(self):
        """Run all remappers.

        All remappers are populated first, then linked, then finalized.
        """
        for remapper in self.remappers.values():
            remapper.populate()
        for remapper in self.remappers.values():
            remapper.link()
        for remapper in self.remappers.values():
            remapper.finalize()

    def __getitem__(self, item):
        return self.remappers[item]


def _grad(mng, root):
    graphs = root.scope

    remappers = RemapperSet(
        graphs,
        grad_fprop=FPropRemapper.partial(),
        grad_fprop_app=FPropAppRemapper.partial(
            master='grad_fprop'
        ),
        grad_bprop=BPropRemapper.partial(
            master='grad_fprop'
        ),
        grad_sens=SensRemapper.partial(
            graph_relation='grad_bprop'
        ),
        grad_bprop_app=BPropAppRemapper.partial(
            master='grad_sens'
        ),
    )
    remappers.run()
    return remappers['grad_fprop'].get_graph(root)


@overload
def J(prim: Primitive, resources):
    """Implement J on a Primitive."""
    g = augmented_graphs[prim]
    if isinstance(g, Graph):
        return clone(g)
    else:
        return g


@overload  # noqa: F811
def J(graph: Graph, manager):
    """Implement J on a Graph."""
    if graph.transforms.get('grad', None):
        return graph.transforms['grad']
    manager.add_graph(graph)
    res = _grad(manager, graph)
    return clone(res)


@overload  # noqa: F811
def J(other: object, manager):
    """We do not implement J on non-functions here."""
    name = type(other).__qualname__
    raise NotImplementedError(f'J(::{name}) not implemented')
