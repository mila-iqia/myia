"""Utilities for manipulating and inspecting the IR."""

from typing import Iterable, Set

from ..graph_utils import (
    EXCLUDE,
    FOLLOW,
    NOFOLLOW,
    dfs as _dfs,
    toposort as _toposort,
)
from ..utils import Var
from .anf import ANFNode, Apply, Constant, Graph, VarNode

#######################
# Successor functions #
#######################


def succ_deep(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    A node's successors are its `incoming` set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if node.is_constant_graph():
        return [node.value.return_] if node.value.return_ else []
    else:
        return node.incoming


def succ_deeper(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    Unlike `succ_deep` this visits all encountered graphs thoroughly, including
    those found through free variables.
    """
    if node.is_constant_graph():
        return [node.value.return_] if node.value.return_ else []
    elif node.graph:
        return list(node.incoming) + [node.graph.return_]
    else:
        return node.incoming


def succ_incoming(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming."""
    return node.incoming


#################################
# Inclusion/exclusion functions #
#################################


def exclude_from_set(stops):
    """Avoid visiting nodes in the stops set."""
    if not isinstance(stops, (set, frozenset, dict)):
        stops = frozenset(stops)

    def include(node):
        return EXCLUDE if node in stops else FOLLOW

    return include


def freevars_boundary(graph, include_boundary=True):
    """Stop visiting when encountering free variables.

    Arguments:
        graph: The main graph from which we want to include nodes.
        include_boundary: Whether to yield the free variables or not.

    """

    def include(node):
        g = node.graph
        if g is None or g is graph:
            return FOLLOW
        elif include_boundary:
            return NOFOLLOW
        else:
            return EXCLUDE

    return include


#####################
# Search algorithms #
#####################


def dfs(root: ANFNode, follow_graph: bool = False) -> Iterable[ANFNode]:
    """Perform a depth-first search."""
    return _dfs(root, succ_deep if follow_graph else succ_incoming)


def toposort(root: ANFNode) -> Iterable[ANFNode]:
    """Order the nodes topologically."""
    return _toposort(root, succ_incoming)


###############
# Isomorphism #
###############


def _same_node_shallow(n1, n2, equiv):
    # Works for Constant, Parameter and nodes previously seen
    if n1 in equiv and equiv[n1] is n2:
        return True
    elif n1.is_constant_graph() and n2.is_constant_graph():
        # Note: we provide current equiv so that nested graphs can properly
        # match their free variables, using the equiv of their parent graph.
        return isomorphic(n1.value, n2.value, equiv)
    elif n1.is_constant():
        return n1.value == n2.value
    elif n1.is_parameter():
        # Parameters are matched together in equiv when we ask whether two
        # graphs are isomorphic. Therefore, we only end up here when trying to
        # match free variables.
        return False
    else:
        raise TypeError(n1)  # pragma: no cover


def _same_node(n1, n2, equiv):
    # Works for Apply (when not seen previously) or other nodes
    if n1.is_apply():
        return all(
            _same_node_shallow(i1, i2, equiv)
            for i1, i2 in zip(n1.inputs, n2.inputs)
        )
    else:
        return _same_node_shallow(n1, n2, equiv)


def _same_subgraph(root1, root2, equiv):
    # Check equivalence between two subgraphs, starting from root1 and root2,
    # using the given equivalence dictionary. This is a modified version of
    # toposort that walks the two graphs in lockstep.

    done: Set = set()
    todo = [(root1, root2)]

    while todo:
        n1, n2 = todo[-1]
        if n1 in done:
            todo.pop()
            continue
        cont = False

        s1 = list(succ_incoming(n1))
        s2 = list(succ_incoming(n2))
        if len(s1) != len(s2):
            return False
        for i, j in zip(s1, s2):
            if i not in done:
                todo.append((i, j))
                cont = True

        if cont:
            continue
        done.add(n1)

        res = _same_node(n1, n2, equiv)
        if res:
            equiv[n1] = n2
        else:
            return False

        todo.pop()

    return True


def isomorphic(g1, g2, equiv=None):
    r"""Return whether g1 and g2 are structurally equivalent.

    Constants are isomorphic iff they contain the same value or are isomorphic
    graphs.

    g1.return\_ and g2.return\_ must represent the same node under the
    isomorphism. Parameters must match in the same order.
    """
    if equiv is None:
        equiv = {}

    if (g1, g2) in equiv:
        return equiv[(g1, g2)] is not False

    if len(g1.parameters) != len(g2.parameters):
        return False

    equiv.update(dict(zip(g1.parameters, g2.parameters)))
    equiv[(g1, g2)] = "PENDING"
    rval = _same_subgraph(g1.return_, g2.return_, equiv)
    equiv[(g1, g2)] = rval

    return rval


########
# Misc #
########


def sexp_to_node(sexp, graph, multigraph=False, sub=None):
    """Convert an s-expression (tuple) to a subgraph.

    Args:
        sexp: A nested tuple that represents an expression.
        graph: The graph in which to place the created nodes.
        multigraph: If multigraph is True and graph is a Var, then
            every child node will have a fresh Var as its Graph.
            In short, use multigraph=True to get a subgraph where
            each node can be in a different graph. Otherwise, all
            nodes are required to belong to the same graph.
        sub: Substitutions to make.

    Returns:
        An ANFNode equivalent to the given s-expression.

    """
    if isinstance(sexp, tuple):
        if multigraph and isinstance(graph, Var):
            return Apply(
                [sexp_to_node(x, Var("G"), True, sub) for x in sexp], graph
            )
        else:
            return Apply(
                [sexp_to_node(x, graph, multigraph, sub) for x in sexp], graph
            )
    elif sub and sexp in sub:
        return sub[sexp]
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


def print_graph(g):
    """Returns a textual representation of a graph."""
    import io

    buf = io.StringIO()
    print(
        f"graph {g.debug.debug_name}("
        + ", ".join(f"%{p.debug.debug_name}" for p in g.parameters)
        + ") {",
        file=buf,
    )

    def repr_node(node):
        if node.is_constant_graph():
            return f"@{node.value.debug.debug_name}"
        elif node.is_constant():
            return str(node.value)
        else:
            return f"%{node.debug.debug_name}"

    for node in toposort(g.output):
        if node.is_apply():
            print(f"  %{node.debug.debug_name} = ", end="", file=buf)
            print(f"{repr_node(node.inputs[0])}(", end="", file=buf)
            print(
                ", ".join(repr_node(a) for a in node.inputs[1:]),
                end="",
                file=buf,
            )
            print(")", file=buf)
        elif node.is_constant():
            pass
        elif node.is_parameter():
            pass
        else:  # pragma: no cover
            print(f"UNK: {node}", file=buf)

    print(f"  return %{g.output.debug.debug_name}", file=buf)
    print("}", file=buf)
    return buf.getvalue()


__all__ = [
    "dfs",
    "exclude_from_set",
    "freevars_boundary",
    "isomorphic",
    "print_graph",
    "sexp_to_graph",
    "sexp_to_node",
    "succ_deep",
    "succ_deeper",
    "succ_incoming",
    "toposort",
]
