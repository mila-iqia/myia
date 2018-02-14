"""
Utilities for closure conversion.

I.e. retrieving nesting structure and listing free variables.
"""


from typing import Dict, Iterable, Optional, Set
from collections import defaultdict

from myia.anf_ir import ANFNode, Constant, Graph
from myia.anf_ir_utils import is_constant_graph, succ_deep, succ_stop_at_fv
from myia.graph_utils import dfs
from myia.utils import memoize_method


class ParentProxy:
    """Represents a graph's immediate parent."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the ParentProxy."""
        self.graph = graph


class NestingAnalyzer:
    """Analyzes the nesting structure of a graph."""

    def __init__(self, root):
        """Initialize a NestingAnalyzer."""
        self.root = root

    @memoize_method
    def coverage(self) -> Iterable[Graph]:
        """Return a collection of graphs accessible from the root."""
        nodes = dfs(Constant(self.root), succ_deep)
        return [node.value for node in nodes
                if is_constant_graph(node)]

    @memoize_method
    def free_variables_direct(self) -> Dict[Graph, Iterable[ANFNode]]:
        """Return a mapping from each graph to its free variables.

        The free variables returned are those that the graph refers
        to directly. Nested graphs are not taken into account, but
        they are in `free_variables_total`.
        """
        coverage = self.coverage()
        return {g: [node for node in dfs(g.return_,
                                         succ_stop_at_fv(g))
                    if node.graph and node.graph is not g]
                for g in coverage}

    @memoize_method
    def graph_dependencies_direct(self) -> Dict[Graph, Set[Graph]]:
        """Map each graph to the graphs it gets free variables from.

        This is the set of graphs that own the nodes returned by
        `free_variables_direct`, for each graph.
        """
        fvdict = self.free_variables_direct()
        return {g: {node.graph for node in fvs}
                for g, fvs in fvdict.items()}

    @memoize_method
    def graphs_used(self) -> Dict[Graph, Set[Graph]]:
        """Map each graph to the set of graphs it uses.

        For each graph, this is the set of graphs that it refers to
        directly.
        """
        coverage = self.coverage()
        return {g: {node.value for node in dfs(g.return_,
                                               succ_stop_at_fv(g))
                    if is_constant_graph(node)}
                for g in coverage}

    @memoize_method
    def graph_dependencies_total(self) -> Dict[Graph, Set[Graph]]:
        """Map each graph to the set of graphs it depends on.

        This is a superset of `graph_dependencies_direct` which also
        includes the graphs from which nested graphs need free
        variables.
        """
        gdir = self.graph_dependencies_direct()
        used = self.graphs_used()
        all_deps = {g: deps | {ParentProxy(g)
                               for g in used.get(g, set())}
                    for g, deps in gdir.items()}

        def seek_parents(g, path=None):
            if path is None:
                path = set()
            if g in path:
                return set()
            deps = all_deps[g]
            parents: Set[Graph] = set()
            for dep in deps:
                if isinstance(dep, ParentProxy):
                    parents |= seek_parents(dep.graph, path | {g})
                else:
                    parents.add(dep)
            return parents - {g}

        new_deps = {}
        for g in all_deps.keys():
            new_deps[g] = seek_parents(g)
        return new_deps

    @memoize_method
    def parents(self) -> Dict[Graph, Optional[Graph]]:
        """Map each graph to its parent graph.

        Top-level graphs are associated to `None` in the returned
        dictionary.
        """
        all_deps = self.graph_dependencies_total()
        all_deps = {g: set(deps) for g, deps in all_deps.items()}

        buckets: Dict[int, Set[Graph]] = defaultdict(set)
        for g, deps in all_deps.items():
            buckets[len(deps)].add(g)

        rm: Set[Graph] = set()
        next_rm: Set[Graph] = set()
        for n, graphs in sorted(buckets.items()):
            for g in graphs:
                all_deps[g] -= rm
                assert len(all_deps[g]) <= 1
                next_rm |= all_deps[g]
            rm |= next_rm

        parents = {g: None if len(gs) == 0 else list(gs)[0]
                   for g, gs in all_deps.items()}

        return parents

    @memoize_method
    def children(self) -> Dict[Graph, Set[Graph]]:
        """Map each graph to the graphs immediately nested in it.

        This is the inverse map of `parents`.
        """
        children: Dict[Graph, Set[Graph]] = defaultdict(set)
        for g, parent in self.parents().items():
            if parent:
                children[parent].add(g)
        return children

    @memoize_method
    def free_variables_total(self) -> Dict[Graph, Set[ANFNode]]:
        """Map each graph to its free variables.

        This differs from `free_variables_direct` in that it also
        includes free variables needed by children graphs.
        Furthermore, graph Constants may figure as free variables.
        """
        parents = self.parents()
        fvs: Dict[Graph, Set[ANFNode]] = defaultdict(set)

        for node in dfs(self.root.return_, succ_deep):
            for inp in node.inputs:
                if is_constant_graph(inp):
                    owner = parents[inp.value]
                else:
                    owner = inp.graph
                if owner is None:
                    continue
                g = node.graph
                while g is not owner:
                    fvs[g].add(inp)
                    g = parents[g]

        return fvs

    def nested_in(self, g1: Graph, g2: Graph) -> bool:
        """Return whether g1 is nested in g2."""
        parents = self.parents()
        while g1:
            g1 = parents[g1]
            if g1 is g2:
                return True
        else:
            return False
