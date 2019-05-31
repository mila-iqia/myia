"""Graph cloning facilities.

GraphRemapper/BasicRemapper are the core remapping classes that can be extended
by graph transformers.
"""

from dataclasses import dataclass
from copy import copy

from ..info import About
from ..utils import Partializable

from .anf import Apply, Constant, Graph, Parameter, ANFNode
from .manager import manage


#################
# Graph cloning #
#################


@dataclass
class _ToLink:
    """Describes a node that we want to link in the link phase."""

    key: object
    graph: Graph
    new_graph: Graph
    node: ANFNode
    new_node: ANFNode


class GraphRemapper(Partializable):
    """Base class for remappers.

    A remapper maps every node of a graph to a new node in a different graph.

    Remapping rules can be adapted in subclasses.

    Arguments:
        graphs: The graphs to transform.
        relation: The relation between the original node and the new node.
        remappers (Optional): A RemapperSet for if the remapped nodes need
            to refer to nodes in other remappers.
        manager: The manager to use, or None to fetch it automatically.
        graph_relation (Optional): The relation between the original graph
            and the new graph (defaults to relation).
    """

    def __init__(self,
                 graphs,
                 *,
                 relation,
                 remappers=None,
                 manager=None,
                 graph_relation=None):
        """Initialize a GraphRemapper."""
        self.relation = relation
        self.graph_relation = graph_relation or relation
        self.graphs = graphs
        if manager is None:
            self.manager = manage(*graphs, weak=True)
        else:
            self.manager = manager
        self.graph_repl = {None: None}
        self.repl = {}
        self.to_link = []
        self.remappers = remappers

    def get_graph(self, graph):
        """Get the new graph corresponding to graph."""
        return self.graph_repl[graph]

    def gen_graph(self, graph):
        """Generate the corresponding new graph."""
        raise NotImplementedError('Override gen_graph in subclass')

    def gen_parameter(self, graph, new_graph, node):
        """Generate the corresponding new parameter."""
        raise NotImplementedError('Override gen_parameter in subclass')

    def gen_apply(self, graph, new_graph, node):
        """Generate the corresponding new application."""
        raise NotImplementedError('Override gen_apply in subclass')

    def gen_constant(self, graph, new_graph, node):
        """Generate the corresponding new constant."""
        raise NotImplementedError('Override gen_constant in subclass')

    def gen_constant_graph(self, graph, new_graph, ct):
        """Generate the corresponding new constant graph."""
        return self.gen_constant(graph, new_graph, ct)

    gen_child = NotImplemented
    gen_fv = NotImplemented
    gen_fv_graph = NotImplemented

    def gen_rogue_parameter(self, graph, new_graph, p):  # pragma: no cover
        """Generate something for a parameter not in the parameter list."""
        raise Exception(f'Found a parameter not in the parameter list: {p}')

    def remap_node(self, key, graph, node, new_graph, new_node, link=None):
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
        if key in self.repl:
            return self.repl[key]
        if link is None:
            link = new_node.is_apply()
        self.repl[key] = new_node
        if link:
            work = _ToLink(
                key=key,
                graph=graph,
                node=node,
                new_graph=new_graph,
                new_node=new_node
            )
            self.to_link.append(work)
        return new_node

    def link_apply(self, link):
        """Generate the inputs for new_node."""
        raise NotImplementedError()

    def finalize_graph(self, graph, new_graph):
        """Finalize the new graph new_graph (set its output)."""
        raise NotImplementedError()

    def generate(self):
        """Generate all the new graphs."""
        for graph in self.graphs:
            self.gen_graph(graph)

    def populate(self):
        """Generate all the new nodes."""
        for graph in self.graphs:
            target_graph = self.get_graph(graph)
            mng = self.manager

            for p in graph.parameters:
                self.gen_parameter(graph, target_graph, p)

            for node in mng.nodes[graph]:
                if node.is_apply():
                    self.gen_apply(graph, target_graph, node)
                elif node.is_parameter() and node not in graph.parameters:
                    self.gen_rogue_parameter(
                        graph, target_graph, node)

            if self.gen_child is not NotImplemented:
                for child in mng.children[graph]:
                    self.gen_child(graph, target_graph, child)

            if self.gen_fv is not NotImplemented:
                for node in mng.free_variables_total[graph]:
                    if isinstance(node, ANFNode):
                        self.gen_fv(graph, target_graph, node)

            if self.gen_fv_graph is not NotImplemented:
                for node in mng.free_variables_total[graph]:
                    if isinstance(node, Graph):
                        self.gen_fv_graph(graph, target_graph, node)

            for ct in mng.constants[graph]:
                if ct.is_constant_graph():
                    self.gen_constant_graph(graph, target_graph, ct)
                else:
                    self.gen_constant(graph, target_graph, ct)

    def link(self):
        """Link all new nodes to their new inputs."""
        for link in self.to_link:
            self.link_apply(link)

    def finalize(self):
        """Finalize the graphs that belong to the remapper."""
        for graph in self.graphs:
            self.finalize_graph(graph, self.get_graph(graph))

    def run(self):
        """Run the remapping."""
        self.generate()
        self.populate()
        self.link()
        self.finalize()


class BasicRemapper(GraphRemapper):
    """Basic Remapper.

    Includes sensible defaults for the generation of graphs, parameters,
    apply and constants.
    """

    def gen_graph(self, graph):
        """Makes an new empty graph."""
        with About(graph.debug, self.graph_relation):
            self.graph_repl[graph] = Graph()

    def gen_parameter(self, graph, new_graph, p):
        """Makes a new parameter."""
        with About(p.debug, self.relation):
            new = new_graph.add_parameter()
            self.remap_node(p, graph, p, new_graph, new)

    def gen_apply(self, graph, new_graph, node):
        """Makes an empty Apply node (to link later)."""
        with About(node.debug, self.relation):
            new = Apply([], new_graph)
            self.remap_node(node, graph, node, new_graph, new)

    def gen_constant(self, graph, new_graph, constant):
        """Makes a copy of the constant with the same value."""
        with About(constant.debug, self.relation):
            new = Constant(constant.value)
            self.remap_node(constant, graph, constant, new_graph, new)


class CloneRemapper(BasicRemapper):
    """Remapper for GraphCloner."""

    def __init__(self,
                 graphs,
                 inlines,
                 manager,
                 relation,
                 graph_relation,
                 clone_constants):
        """Initialize the GraphCloner."""
        super().__init__(
            graphs,
            manager=manager or manage(*graphs, *inlines, weak=True),
            relation=relation,
            graph_relation=graph_relation
        )
        self.inlines = inlines
        self.clone_constants = clone_constants

    def remap_node(self, key, graph, node, new_graph, new_node, link=None):
        """Remap the given node as normal and also copy the abstract."""
        nn = super().remap_node(key, graph, node, new_graph, new_node, link)
        nn.abstract = node.abstract
        return nn

    def get_graph(self, graph):
        """Return the graph associated to this graph."""
        if graph in self.inlines:
            target_graph, _ = self.inlines[graph]
        else:
            target_graph = self.graph_repl[graph]
        return target_graph

    def gen_graph(self, graph):
        """Generate a new graph unless the graph is meant to be inlined.

        Copies flags and transforms.
        """
        if graph in self.inlines:
            target_graph, new_params = self.inlines[graph]
            for p, new_p in zip(graph.parameters, new_params):
                self.repl[p] = new_p
        else:
            with About(graph.debug, self.graph_relation):
                target_graph = Graph()
                target_graph.flags = copy(graph.flags)
                target_graph.transforms = copy(graph.transforms)
            self.graph_repl[graph] = target_graph

    def gen_rogue_parameter(self, graph, new_graph, node):
        """Generate a new parameter."""
        # This should not happen for valid graphs, but clone is also used for
        # debugging to if we can avoid failing that is good.
        with About(node.debug, self.relation):
            new = Parameter(new_graph)
            self.remap_node(node, graph, node, new_graph, new)

    def gen_parameter(self, graph, new_graph, p):
        """Generate a new parameter."""
        if graph not in self.inlines:
            super().gen_parameter(graph, new_graph, p)

    def gen_constant(self, graph, new_graph, constant):
        """Generate a new constant if clone_constant is True."""
        if self.clone_constants:
            super().gen_constant(graph, new_graph, constant)

    def gen_constant_graph(self, graph, new_graph, constant):
        """Generate a constant for the cloned graph when applicable."""
        g = constant.value
        if g not in self.inlines and g in self.graph_repl:
            target_graph = self.get_graph(g)
            with About(constant.debug, self.relation):
                new = Constant(target_graph)
                self.remap_node(constant, graph, constant, new_graph, new)

    def link_apply(self, link):
        """Fill a node's inputs."""
        for inp in link.node.inputs:
            repl = self.repl.get((link.graph, inp), None)
            if repl is None:
                repl = self.repl.get(inp, inp)
            link.new_node.inputs.append(repl)

    def finalize_graph(self, graph, new_graph):
        """Set the graph's output, unless it is an inlined graph."""
        if graph not in self.inlines:
            new_graph.return_ = self.repl[graph.return_]

    def clone_disconnected(self, droot):
        """Clone a subgraph that's not (yet) connected to its graph/manager."""
        if droot.graph not in self.graph_repl:
            return droot
        if droot in self.repl:
            return False
        target_graph = self.get_graph(droot.graph)
        if droot.is_parameter() and droot not in self.repl:  # pragma: no cover
            self.gen_rogue_parameter(None, target_graph, droot)
        elif droot.is_apply():
            self.gen_apply(None, target_graph, droot)
            new = self.repl[droot]
            new_inputs = []
            for inp in droot.inputs:
                new_inputs.append(self.clone_disconnected(inp))
            new.inputs = new_inputs
        else:
            return False


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

        All remappers generate their graphs first, then the nodes are
        populated first, then linked, then their graphs are finalized.
        """
        remappers = self.remappers.values()
        for remapper in remappers:
            remapper.generate()
        for remapper in remappers:
            remapper.populate()
        for remapper in remappers:
            remapper.link()
        for remapper in remappers:
            remapper.finalize()

    def __getitem__(self, item):
        """Get a remapper from its name."""
        return self.remappers[item]


class GraphCloner:
    """Facility to clone graphs.

    To get the clone of a graph, index the GraphCloner with the graph,
    e.g. `cloned_g = graph_cloner[g]`

    Arguments:
        graphs: A set of graphs to clone.
        inline: A list of graphs to inline. Each entry must be a triple
            of (graph, target_graph, new_params):
            graph: The original graph, which we want to clone. Graphs
                nested in the original graph will be automatically cloned.
            target_graph: The graph in which to inline.
            new_params: A list of nodes to replace the graph's parameters,
                which usually belong to target_graph or its parents.
        total: Whether to clone every graph encountered when walking
            through the graphs to clone, even if these graphs are
            not nested.
        clone_constants: Whether to clone Constant nodes. Defaults to False.
        clone_children: Whether to clone a graph's full scope. Defaults to
            True.
        relation: The relation the cloned nodes present with respect
            to the originals, for debugging purposes. Default is
            'copy'.
        graph_relation: The relation the cloned graphs present with
            respect to the originals, for debugging purposes. Default
            is the value of the `relation` argument.
    """

    def __init__(self,
                 *graphs,
                 inline=[],
                 total=False,
                 relation='copy',
                 clone_constants=False,
                 clone_children=True,
                 graph_relation=None,
                 remapper_class=CloneRemapper):
        """Initialize a GraphCloner."""
        self.total = total
        self.clone_children = clone_children
        if isinstance(inline, tuple):
            inline = [inline]
        _graphs = graphs + tuple(g for g, _, _ in inline)
        self.manager = manage(*_graphs, weak=True)
        self.collect_graphs(graphs, inline)
        self.remapper = remapper_class(
            graphs=self.graphs,
            manager=self.manager,
            inlines=self.inlines,
            relation=relation,
            graph_relation=graph_relation,
            clone_constants=clone_constants,
        )
        self.remapper.run()

    def collect_graphs(self, graphs, inlines):
        """Collect the full set of graphs to clone.

        This set will include the scopes of the graphs if clone_constants
        is True, as well as any graphs they use if total is True.
        """
        def expand_clones(graph):
            if self.clone_children:
                self.graphs.update(self.manager.scopes[graph] - {graph})
            if self.total:
                self.graphs.update(self.manager.graphs_reachable[graph])

        self.graphs = set()
        self.inlines = {}
        for graph in graphs:
            self.graphs.add(graph)
            expand_clones(graph)
        for graph, target_graph, new_params in inlines:
            self.inlines[graph] = (target_graph, new_params)
            expand_clones(graph)
        if set(self.inlines) & self.graphs:
            msg = 'Trying to clone and inline a graph at the same time.'
            if self.total:
                msg += ' Try setting the `total` option to False.'
            raise Exception(msg)
        self.graphs.update(self.inlines)

    def __getitem__(self, x):
        """Get the clone of the given graph or node."""
        if isinstance(x, Graph):
            return self.remapper.graph_repl.get(x, x)
        else:
            return self.remapper.repl.get(x, x)


def clone(g,
          total=True,
          relation='copy',
          clone_constants=False,
          graph_relation=None):
    """Return a clone of g."""
    return GraphCloner(g,
                       total=total,
                       relation=relation,
                       clone_constants=clone_constants,
                       graph_relation=graph_relation)[g]


def transformable_clone(graph, relation='transform'):
    """Return a clone of the graph that can be safely transformed.

    If the graph is recursive, recursive calls will point to the original
    graph and not to the clone. This allows us to modify the returned graph
    safely, without messing up the recursive call sites.
    """
    with About(graph.debug, relation):
        newg = Graph()
    for p in graph.parameters:
        with About(p.debug, 'copy'):
            newg.add_parameter()
    cl = GraphCloner(inline=(graph, newg, newg.parameters))
    newg.output = cl[graph.output]
    return newg
