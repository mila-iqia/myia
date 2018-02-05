"""
Utilities for closure conversion.

I.e. retrieving nesting structure and listing free variables.
"""


from typing import Set, Dict, Optional, Union
from collections import defaultdict
from .anf_ir import ANFNode, Constant, Graph
from .anf_ir_utils import dfs, is_constant_graph


class ParentProxy:
    """Represents a graph's immediate parent."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the ParentProxy."""
        self.graph = graph


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

    def __init__(self) -> None:
        """Initialize the NestingAnalyzer."""
        self.processed: Set[Graph] = set()
        self.parents: Dict[Graph, Optional[Graph]] = {}
        self.children: Dict[Graph, Set[Graph]] = defaultdict(set)
        self.fvs: Dict[Graph, Set[ANFNode]] = defaultdict(set)

    def run(self, root: Graph) -> 'NestingAnalyzer':
        """Run the analysis. Populates `self.parents` and `self.fvs`."""
        if root in self.processed:
            return self

        # Initialize temporary structures
        self.prox: Dict[Graph, ParentProxy] = {}
        self.graphs: Set[Graph] = set()
        self.nodes: Set[ANFNode] = set()
        self.deps: Dict[Graph, Set[Union[Graph, ParentProxy]]] = \
            defaultdict(set)

        # We get all nodes that could possibly be executed within this
        # graph.
        self.nodes = set(dfs(Constant(root), True))

        # We get all graphs possibly accessed.
        self.graphs = {node.value for node in self.nodes
                       if is_constant_graph(node)}
        self.processed |= self.graphs

        # We compute the dependencies:
        # * If a node in graph g has an input from graph g', then g depends
        #   on g'
        # * If a node in graph g refers to a graph g', then g depends on
        #   whatever the parent scope of g' is. That parent scope is
        #   represented by a ParentProxy.
        self._compute_deps()

        # We complete the dependency graph by following proxies.
        self._resolve_proxies()

        # We remove all but the most immediate parent of a graph.
        self._simplify()

        # We update the parents. ParentProxy instances were removed, so
        # the typing's correct.
        new_parents = {g: None if len(gs) == 0 else list(gs)[0]
                       for g, gs in self.deps.items()}
        self.parents.update(new_parents)  # type: ignore

        # We update the children
        for g, parent in new_parents.items():
            if parent:
                assert isinstance(parent, Graph)
                self.children[parent].add(g)

        # We find all free variables (could be interleaved with other
        # phases, probably)
        self._associate_free_variables()

        return self

    def _compute_deps(self) -> None:
        """Compute which graphs depend on which other graphs."""
        for g in self.graphs:
            prox = ParentProxy(g)
            self.prox[g] = prox
        for node in self.nodes:
            orig = node.graph
            for inp in node.inputs:
                if is_constant_graph(inp):
                    self.deps[orig].add(self.prox[inp.value])
                elif inp.graph and inp.graph is not orig:
                    self.deps[orig].add(inp.graph)

    def _resolve_proxies(self) -> None:
        """Resolve the ParentProxy nodes."""
        def seek_parents(g, path=None):
            if path is None:
                path = set()
            if g in path:
                return set()
            deps = self.deps[g]
            parents = set()
            for dep in deps:
                if isinstance(dep, ParentProxy):
                    parents |= seek_parents(dep.graph, path | {g})
                else:
                    parents.add(dep)
            return parents - {g}

        newdeps = {}
        for g in self.graphs:
            newdeps[g] = seek_parents(g)
        self.deps = newdeps

    def _simplify(self) -> None:
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
            self._simplify()

    def _associate_free_variables(self) -> None:
        """Associate a set of free variables to each graph."""
        for node in self.nodes:
            for inp in node.inputs:
                if is_constant_graph(inp):
                    owner = self.parents[inp.value]
                else:
                    owner = inp.graph
                if owner is None:
                    continue
                g = node.graph
                while g is not owner:
                    self.fvs[g].add(inp)
                    g = self.parents[g]

    def nested_in(self, g1: Graph, g2: Graph) -> bool:
        """Return whether g1 is nested in g2."""
        while g1:
            g1 = self.parents[g1]
            if g1 is g2:
                return True
        else:
            return False
