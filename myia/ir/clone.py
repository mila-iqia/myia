"""Graph cloning facility."""

from typing import Any, Dict, Iterable, Set, Union, cast

from ..cconv import NestingAnalyzer
from ..graph_utils import dfs
from ..info import About
from ..utils import smap

from .anf import ANFNode, Apply, Constant, Graph, Parameter
from .utils import exclude_from_set, is_constant, is_parameter, succ_incoming

#################
# Graph cloning #
#################


class GraphCloner:
    """Facility to clone graphs.

    Graphs to clone can be passed in the constructor or they can be
    added with the `add_clone` method, which offers more options.

    Cloning a graph automatically involves cloning every graph which
    is nested in it.

    To get the clone of a graph, index the GraphCloner with the graph,
    e.g. `cloned_g = graph_cloner[g]`

    Attributes:
        total: Whether to clone every graph encountered when walking
            through the graphs to clone, even if these graphs are
            not nested.
        clone_constants: Whether to clone Constant nodes.
        relation: The relation the cloned nodes present with respect
            to the originals, for debugging purposes. Default is
            'copy'.

    """

    def __init__(self,
                 *graphs: Graph,
                 total: bool = False,
                 clone_constants: bool = False,
                 relation: str = 'copy') -> None:
        """Initialize a GraphCloner."""
        self.todo: Set[Graph] = set()
        # graph_clones maps a graph to either:
        # * The clone of this graph
        # * The string 'inline', if the graph was inlined
        self.graph_clones: Dict[Graph, Union[Graph, str]] = {}
        # repl maps:
        # * A node to its clone
        # * A graph to either its clone or the graph in which to inline it
        self.repl: Dict[Union[Graph, ANFNode],
                        Union[Graph, ANFNode]] = {None: None}
        self.total = total
        self.clone_constants = clone_constants
        self.relation = relation
        for graph in graphs:
            self.add_clone(graph)

    def _add_clone(self,
                   graph: Graph,
                   target_graph: Graph = None,
                   set_output: bool = None) -> Graph:
        """Register a clone for the given graph.

        This does not register graphs nested in the given graph, so
        use this method with caution.

        Arguments:
            graph: The graph to clone.
            target_graph: The graph which will contain the clones of
                the nodes in the original graph. If target_graph is
                None or not given, a new Graph will be created to
                serve as the target_graph.
            set_output: Whether the output of the target_graph should
                be set to the clone of the output of the original
                graph. If set_output is None or not given, then it
                is true if target_graph is None (and thus must be
                created), false otherwise.

        Return:
            target_graph, or the clone associated to the original graph.
        """
        if set_output is None:
            set_output = target_graph is None

        if target_graph is None:
            with About(graph.debug, self.relation):
                target_graph = Graph()

        self.repl[graph] = target_graph
        if set_output:
            self.graph_clones[graph] = target_graph
        else:
            self.graph_clones[graph] = 'inline'
        self.todo.add(graph)

        return target_graph

    def add_clone(self,
                  graph: Graph,
                  target_graph: Graph = None,
                  new_params: Iterable[ANFNode] = None,
                  set_output: bool = None):
        """Add a clone for the given graph.

        This method can also be used for inlining.

        Any graph nested in the given graph will also be cloned.

        Arguments:
            graph: The original graph, which we want to clone. Graphs
                nested in the original graph will be automatically cloned.
            target_graph: The graph in which to place clones of the nodes
                in the original graph. If target_graph is None or not given,
                a fresh Graph will be created automatically. If it is a
                Graph, the original graph will be essentially inlined
                in the target_graph.
            new_params: A list of nodes to serve as the new parameters of
                the cloned graph, if the goal is inlining.
            set_output: (Advanced option) Whether the output of the
                target_graph should
                be set to the clone of the output of the original
                graph. If set_output is None or not given, then it
                is true if target_graph is None (and thus must be
                created), false otherwise (because the intent is most
                likely inlining into target_graph, and you don't want
                to set the output in this case).
        """
        nest = NestingAnalyzer(graph)
        nested_graphs = nest.scopes()[graph]
        for g in nested_graphs:
            if g is not graph:
                self._add_clone(g)

        self._add_clone(graph, target_graph, set_output)

        if new_params:
            for p, new_p in zip(graph.parameters, new_params):
                self.repl[p] = new_p

    def _is_inline(self, g: Graph) -> bool:
        return self.graph_clones.get(g, None) == 'inline'

    def _get_graph_replacement(self, g: Graph) -> Graph:
        if self.total and g and g not in self.graph_clones:
            # When the total option is given, we clone any new graph
            # we encounter.
            self._add_clone(g)
        # When the total option is not given, we do not clone graphs
        # unless specified explicitly.
        g2 = self.repl.get(g, g)
        return cast(Graph, g2)

    def _clone_subgraph(self, root: ANFNode) -> ANFNode:
        """Helper function to clone starting from root.

        Arguments:
            root: The starting point for the cloning.

        Return: The clone of root.
        """
        to_clone = list(dfs(root,
                            succ_incoming,
                            exclude_from_set(self.repl)))

        for node in to_clone:
            assert node not in self.repl
            g = self._get_graph_replacement(node.graph)
            if g and g is node.graph:
                self.repl[node] = node
                continue

            new: ANFNode
            with About(node.debug, self.relation):
                if is_parameter(node):
                    new = Parameter(g)
                elif is_constant(node):
                    def convert(x):
                        if isinstance(x, Graph):
                            # We must change the constant to point to the clone
                            # of the graph x, if x was cloned.
                            if self._is_inline(x):
                                # We can either clone a graph or inline it, but
                                # we can't do both.
                                if self.total:
                                    raise Exception(
                                        '`total` option is not compatible'
                                        ' with inlining.'
                                    )
                                else:
                                    return x
                            return self._get_graph_replacement(x)
                        else:
                            return x
                    # This will also properly handle e.g. tuples of graphs
                    new_value = smap(convert, node.value)
                    if new_value is node.value and not self.clone_constants:
                        new = node
                    else:
                        new = Constant(new_value)
                else:
                    new = Apply([], g)
            self.repl[node] = new

        for _node in to_clone:
            node = cast(ANFNode, _node)
            new_inputs = [self.repl[orig_node] for orig_node in node.inputs]
            repl = cast(ANFNode, self.repl[node])
            repl.inputs = new_inputs  # type: ignore

        return cast(ANFNode, self.repl[root])

    def run(self) -> None:
        """Clone everything still to be cloned.

        It is not necessary to run this method, because `getitem` runs
        it automatically.
        """
        while self.todo:
            graph = self.todo.pop()
            new_graph = self.graph_clones[graph]
            root = graph.output
            new_root = self._clone_subgraph(root)
            if new_graph is not 'inline':
                assert isinstance(new_graph, Graph)
                new_graph.output = new_root
                assert all(is_parameter(p) for p in graph.parameters)
                new_graph.parameters = [cast(Parameter, self.repl[p])
                                        for p in graph.parameters]

    def __getitem__(self, x: Any) -> Any:
        """Get the clone of the given graph or node."""
        self.run()
        if self._is_inline(x):
            # Inlined graphs don't have their own clones, since they are
            # inserted in other graphs.
            return x
        else:
            return self.repl.get(x, x)
