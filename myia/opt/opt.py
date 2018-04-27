"""Graph optimization routines."""


from ..graph_utils import dfs, toposort
from ..ir import ANFNode, Apply, Constant, Graph, Special, \
    freevars_boundary, succ_incoming, is_constant_graph, replace
from ..unify import Unification, Var


class VarNode(Special):
    """Graph node that represents a variable."""

    @property
    def __var__(self):
        return self.special


def sexp_to_node(sexp, graph, multigraph=False):
    """Convert an s-expression (tuple) to a subgraph.

    Args:
        sexp: A nested tuple that represents an expression.
        graph: The graph in which to place the created nodes.
        multigraph: If multigraph is True and graph is a Var, then
            every child node will have a fresh Var as its Graph.
            In short, use multigraph=True to get a subgraph where
            each node can be in a different graph. Otherwise, all
            nodes are required to belong to the same graph.

    Returns:
        An ANFNode equivalent to the given s-expression.

    """
    if isinstance(sexp, tuple):
        if multigraph and isinstance(graph, Var):
            return Apply([sexp_to_node(x, Var('G'), True)  # type: ignore
                          for x in sexp], graph)  # type: ignore
        else:
            return Apply([sexp_to_node(x, graph, multigraph)
                          for x in sexp], graph)
    elif isinstance(sexp, Var):
        return VarNode(sexp, graph)
    elif isinstance(sexp, ANFNode):
        return sexp
    else:
        return Constant(sexp)


def sexp_to_graph(sexp):
    """Convert an s-expression to a Graph.

    This converts the s-expression to a subgraph of ANFNodes and then sets that
    subgraph as the output of a new Graph.
    """
    g = Graph()
    g.output = sexp_to_node(sexp, g)
    return g


class PatternSubstitutionOptimization:
    """An optimization that replaces one pattern by another.

    Args:
        pattern: An s-expression, represented as a nested tuple, that
            represents an expression to match. Terms in the s-expression
            may be Var instances or constants.
        replacement:
            * An s-expression, represented as a nested tuple, to
              instantiate to replace the pattern. Vars in the pattern can
              be reused in the replacement.
            * OR a function which will be called when the pattern is
              matched, with the node and the equivalence dictionary as
              arguments.
        name: The name of the optimization.
        multigraph: Whether the pattern can span multiple graphs or not.
            A pattern can span multiple graphs if, for example, the root
            of the pattern is in a closure, and some of the leaves are in
            the parent function of that closure.

    Attributes:
        pattern: The pattern, converted to Myia's IR.
        replacement: The replacement, converted to Myia's IR.
        name: The name of the optimization.

    """

    def __init__(self,
                 pattern,
                 replacement,
                 *,
                 name=None,
                 multigraph=True):
        """Initialize va PatternSubstitutionOptimization."""
        g: Var = Var('RootG')
        self.pattern = sexp_to_node(pattern, g, multigraph)
        if callable(replacement):
            self.replacement = replacement
        else:
            self.replacement = sexp_to_node(replacement, g)
        self.unif = Unification()
        self.name = name

    def __call__(self, node):
        """Return a replacement for the node, if the pattern matches.

        The replacement will be instantiated in the graph of the root of the
        pattern, except for matched nodes in the pattern, which are kept
        unchanged in the replacement.

        Returns:
            * None if the pattern does not match.
            * A subgraph for the reification of the replacement, with
              variables filled in, if the pattern matches.

        """
        equiv = self.unif.unify(node, self.pattern)
        if equiv:
            if callable(self.replacement):
                return self.replacement(node, equiv)
            else:
                return self.unif.reify(self.replacement, equiv)
        else:
            return None


def pattern_replacer(*pattern):
    """Create a PatternSubstitutionOptimization using this function."""
    def deco(f):
        return PatternSubstitutionOptimization(pattern, f, name=f.__name__)
    return deco


class PatternOptimizerSinglePass:
    """Single optimization pass using the given patterns.

    Args:
        patterns: A set of patterns to apply to each node in a graph.

    """

    def __init__(self, patterns):
        """Initialize the PatternOptimizerSinglePass."""
        self.patterns = patterns

    def iterate(self, graph):
        """Iterate through the nodes of the graph.

        The iterator proceeds in topological order.
        """
        incl = freevars_boundary(graph, False)
        topo = toposort(graph.output, succ_incoming, incl)
        return topo

    def replace(self, old, new):
        """Replace a node by another."""
        return replace(old, new)

    def __call__(self, graph):
        """Apply all patterns to all nodes of the given graph."""
        changes = False
        for node in self.iterate(graph):
            for pattern in self.patterns:
                new = pattern(node)
                if new and new is not node:
                    self.replace(node, new)
                    changes = True
                    continue
        return changes


class PatternOptimizerEquilibrium:
    """Run an optimization pass until equilibrium.

    Args:
        single_pass: An optimization pass on a graph.
        auto_acquire: Whether to process any new graph encountered
            when processing a graph.

    """

    def __init__(self, single_pass, *, auto_acquire=True):
        """Initialize a PatternOptimizerEquilibrium."""
        self.single_pass = single_pass
        self.auto_acquire = auto_acquire

    def __call__(self, *graphs):
        """Apply the pass on all graphs repeatedly until equilibrium."""
        graphs = set(graphs)  # type: ignore
        any_changes = 0

        changes = 1
        while changes:
            new_graphs = set()
            changes = 0
            for graph in graphs:
                chg = self.single_pass(graph)
                if chg and self.auto_acquire:
                    for node in dfs(graph.output, succ_incoming):
                        if is_constant_graph(node):
                            new_graphs.add(node.value)
                changes |= chg
                any_changes |= changes
            graphs |= new_graphs  # type: ignore

        return any_changes
