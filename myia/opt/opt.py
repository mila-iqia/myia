"""Graph optimization routines."""

from collections import defaultdict

from ..graph_utils import dfs
from ..ir import ANFNode, Apply, Constant, Graph, Special, \
    succ_deeper, is_constant_graph, replace, is_apply, \
    GraphCloner, manage
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
            return Apply([sexp_to_node(x, Var('G'), True)
                          for x in sexp], graph)
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


class PatternEquilibriumOptimizer:
    """Apply a set of local pattern optimizations until equilibrium."""

    def __init__(self, *node_transformers):
        """Initialize a PatternEquilibriumOptimizer."""
        self.node_transformers = node_transformers

    def __call__(self, *graphs):
        """Apply optimizations until equilibrium on given graphs."""
        mng = manage(*graphs)
        while True:
            changes = False

            for node in mng.all_nodes:
                for transformer in self.node_transformers:
                    new = transformer(node)
                    if new and new is not node:
                        new.type = node.type
                        mng.push_replace(node, new)
                        changes = True
                        break

            mng.commit()

            if not changes:
                break


def inline_unique_uses(graph):
    """Inline every graph that is only used once."""
    graph_uses = defaultdict(set)
    for node in dfs(graph.return_, succ_deeper):
        if is_apply(node):
            for i, inp in enumerate(node.inputs):
                if is_constant_graph(inp):
                    graph_uses[inp.value].add((node, i))

    cloneseq = []

    def lookup(x):
        for c in cloneseq:
            x = c[x]
        return x

    for g, uses in graph_uses.items():
        if len(uses) != 1:
            continue
        (node, i), = uses
        if i != 0:
            continue

        g = lookup(g)
        node = lookup(node)

        clone = GraphCloner(total=False)
        clone.add_clone(g, node.graph, node.inputs[1:])
        cloneseq.append(clone)
        replace(node, clone[g.output])
