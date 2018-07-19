"""Graph cloning facility."""

from copy import copy

from .anf import Apply, Constant, Graph
from ..info import About
from .manager import manage


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
        graph_relation: The relation the cloned graphs present with
            respect to the originals, for debugging purposes. Default
            is the value of the `relation` argument.

    """

    def __init__(self,
                 *graphs,
                 total=False,
                 relation='copy',
                 clone_constants=False,
                 graph_relation=None):
        """Initialize a GraphCloner."""
        self.todo = []
        self.status = {}
        self.nodes = []
        self.clone_constants = clone_constants
        # repl maps a node or a graph to its clone
        self.repl = {None: None}
        self.total = total
        self.relation = relation
        self.graph_relation = graph_relation or relation
        for graph in graphs:
            self.add_clone(graph)

    def add_clone(self, graph, target_graph=None, new_params=None):
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
        """
        self.todo.append((graph, target_graph, new_params))

    def _process_graph(self, graph, target_graph, new_params):
        mng = self.manager

        inline = target_graph is not None

        status = self.status.get(graph, None)
        if status is not None:
            if status == inline:
                return
            else:
                msg = 'Trying to clone and inline a graph at the same time.'
                if self.total:
                    msg += ' Try setting the `total` option to False.'
                raise Exception(msg)

        if inline:
            for p, new_p in zip(graph.parameters, new_params):
                self.repl[p] = new_p

        else:
            with About(graph.debug, self.graph_relation):
                target_graph = Graph()
            for p in graph.parameters:
                with About(p.debug, self.relation):
                    p2 = target_graph.add_parameter()
                    p2.inferred = copy(p.inferred)
                    self.repl[p] = p2
            self.repl[graph] = target_graph

        for node in mng.nodes[graph]:
            if node in self.repl:
                continue
            with About(node.debug, self.relation):
                new = Apply([], target_graph)
                new.inferred = copy(node.inferred)
                self.repl[node] = new
                self.nodes.append((node, new))

        if not inline:
            target_graph.return_ = self.repl[graph.return_]
            for ct in mng.graph_constants[graph]:
                with About(ct.debug, self.relation):
                    new = Constant(target_graph)
                    new.inferred = copy(ct.inferred)
                    self.repl[ct] = new

        if self.clone_constants:
            for ct in mng.constants[graph]:
                if ct not in self.repl:
                    new = Constant(ct.value)
                    new.inferred = copy(ct.inferred)
                    self.repl[ct] = new

        self.status[graph] = inline

        self.todo += [(g, None, None)
                      for g in mng.scopes[graph]
                      if g is not graph]
        if self.total:
            self.todo += [(g, None, None)
                          for g in mng.graphs_used[graph]]

    def run(self):
        """Clone everything still to be cloned.

        It is not necessary to run this method, because `getitem` runs
        it automatically.
        """
        todo = self.todo

        if not todo:
            return

        self.manager = manage(*[g for g, _, _ in todo], weak=True)

        while todo:
            item = todo.pop()
            self._process_graph(*item)

        for old_node, new_node in self.nodes:
            for inp in old_node.inputs:
                repl = self.repl.get(inp, inp)
                new_node.inputs.append(repl)

    def __getitem__(self, x):
        """Get the clone of the given graph or node."""
        self.run()
        return self.repl.get(x, x)


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
