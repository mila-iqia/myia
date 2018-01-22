"""
Utilities for closure conversion.

I.e. retrieving nesting structure and listing free variables.
"""


from collections import defaultdict
from .anf_ir import Constant, Graph
from .anf_ir_utils import dfs


def is_graph(x):
    """Return whether x is a Constant with a Graph value."""
    return isinstance(x, Constant) and isinstance(x.value, Graph)


class NestingAnalyzer:
    """
    Analyzes the nesting structure of a graph.

    Call the `run(root_graph)` method to populate the data structures.

    Attributes:
        deps: Maps a graph to the set of graphs that it depends on.
        parents: Maps a graph to its immediate parent or `None` if
            it has no parent.
        fvs: Maps a graph to a set of nodes it refers to from other
            graphs, including nodes nested graphs have to access.

    """

    class ParentProxy:
        """Represents a graph's immediate parent."""

        def __init__(self, graph):
            """Initialize the ParentProxy."""
            self.graph = graph

    def __init__(self):
        """Initialize the NestingAnalyzer."""
        self.processed = set()
        self.parents = {}
        self.fvs = defaultdict(set)

    def run(self, root):
        """Run the analysis. Populates `self.parents` and `self.fvs`."""
        if root in self.processed:
            return self

        # Initialize temporary structures
        self.prox = {}
        self.graphs = set()
        self.nodes = set()
        self.deps = defaultdict(set)

        # We get all nodes that could possibly be executed within this
        # graph.
        self.nodes = set(dfs(Constant(root), True))

        # We get all graphs possibly accessed.
        self.graphs = {node.value for node in self.nodes if is_graph(node)}
        self.processed |= self.graphs

        # We compute the dependencies:
        # * If a node in graph g has an input from graph g', then g depends
        #   on g'
        # * If a node in graph g refers to a graph g', then g depends on
        #   whatever the parent scope of g' is. That parent scope is
        #   represented by a ParentProxy.
        self.compute_deps()

        # We complete the dependency graph by following proxies.
        self.resolve_proxies()

        # We remove all but the most immediate parent of a graph.
        self.simplify()

        # We update the parents
        new_parents = {g: None if len(gs) == 0 else list(gs)[0]
                       for g, gs in self.deps.items()}
        self.parents.update(new_parents)

        # We find all free variables (could be interleaved with other
        # phases, probably)
        self.associate_free_variables()

        return self

    def compute_deps(self):
        """Compute which graphs depend on which other graphs."""
        for g in self.graphs:
            prox = self.ParentProxy(g)
            self.prox[g] = prox
        for node in self.nodes:
            orig = node.graph
            for inp in node.inputs:
                if is_graph(inp):
                    self.deps[orig].add(self.prox[inp.value])
                elif inp.graph and inp.graph is not orig:
                    self.deps[orig].add(inp.graph)

    def resolve_proxies(self):
        """Resolve the ParentProxy nodes."""
        def seek_parents(g, path=None):
            if path is None:
                path = set()
            if g in path:
                return set()
            deps = self.deps[g]
            parents = set()
            for dep in deps:
                if isinstance(dep, self.ParentProxy):
                    parents |= seek_parents(dep.graph, path | {g})
                else:
                    parents.add(dep)
            return parents - {g}

        newdeps = {}
        for g in self.graphs:
            newdeps[g] = seek_parents(g)
        self.deps = newdeps

    def simplify(self):
        """Keep only the closest dependency for each graph."""
        to_cull = {list(deps)[0]
                   for g, deps in self.deps.items()
                   if len(deps) == 1}

        done = True
        for g, deps in self.deps.items():
            if len(deps) > 1:
                deps.difference_update(to_cull)
                done = False

        if not done:
            self.simplify()

    def associate_free_variables(self):
        """Associate a set of free variables to each graph."""
        for node in self.nodes:
            for inp in node.inputs:
                if is_graph(inp):
                    owner = self.parents[inp.value]
                else:
                    owner = inp.graph
                if owner is None:
                    continue
                g = node.graph
                while g is not owner:
                    self.fvs[g].add(inp)
                    g = self.parents[g]
