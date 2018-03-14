"""Utilities for manipulating and inspecting the IR."""
from typing import Any, Iterable, Callable, Set

from myia.anf_ir import ANFNode, Apply, Constant, Graph, Parameter
<<<<<<< HEAD
from myia.graph_utils import dfs as _dfs, toposort as _toposort, \
    FOLLOW, NOFOLLOW, EXCLUDE
=======
from myia.graph_utils import dfs as _dfs, toposort as _toposort
>>>>>>> Add include argument to dfs, inclusion functions, helpers


#######################
# Successor functions #
#######################


def succ_deep(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    A node's successors are its `incoming` set, or the return node of a graph
    when a graph Constant is encountered.
    """
    if is_constant_graph(node):
        return [node.value.return_] if node.value.return_ else []
    else:
        return node.incoming


def succ_deeper(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming and graph references.

    Unlike `succ_deep` this visits all encountered graphs thoroughly, including
    those found through free variables.
    """
    if is_constant_graph(node):
        return [node.value.return_] if node.value.return_ else []
    elif node.graph:
        return list(node.incoming) + [node.graph.return_]
    else:
        return node.incoming


def succ_incoming(node: ANFNode) -> Iterable[ANFNode]:
    """Follow node.incoming."""
    return node.incoming


def succ_bidirectional(scope: Set[Graph]) -> Callable:
    """Follow node.incoming, node.users and graph references.

    `succ_bidirectional` will only return nodes that belong to the given
    set of graphs.
    """
    def succ(node: ANFNode) -> Iterable[ANFNode]:
        rval = set(node.incoming) | {u for u, _ in node.uses}
        if is_constant_graph(node):
            rval.add(node.value.return_)
        return {x for x in rval if x.graph in scope}

    return succ


#################################
# Inclusion/exclusion functions #
#################################


def exclude_from_set(stops):
    """Avoid visiting nodes in the stops set."""
    if not isinstance(stops, (set, frozenset, dict)):
        stops = frozenset(stops)

    def include(node):
<<<<<<< HEAD
        return EXCLUDE if node in stops else FOLLOW
=======
        return node not in stops
>>>>>>> Add include argument to dfs, inclusion functions, helpers

    return include


def freevars_boundary(graph, include_boundary=True):
    """Stop visiting when encountering free variables.

    Arguments:
        graph: The main graph from which we want to include nodes.
        include_boundary: Whether to yield the free variables or not.
    """
<<<<<<< HEAD
    def include(node):
        g = node.graph
        if g is None or g is graph:
            return FOLLOW
        elif include_boundary:
            return NOFOLLOW
        else:
            return EXCLUDE
=======

    def include(node):
        g = node.graph
        if g is None or g is graph:
            return True
        elif include_boundary:
            return None
        else:
            return False
>>>>>>> Add include argument to dfs, inclusion functions, helpers

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


def accessible_graphs(root: Graph) -> Set[Graph]:
    """Return all Graphs accessible from root."""
    return {root} | {x.value for x in dfs(root.return_, True)
                     if is_constant_graph(x)}


###########
# Cleanup #
###########


def destroy_disconnected_nodes(root: Graph) -> None:
    """Remove dead nodes that belong to the graphs accessible from root.

    The `uses` set of a node may keep alive some nodes that are not connected
    to the output of a graph (e.g. `_, x = pair`, where `_` is unused). These
    nodes are removed by this function.
    """
    # We restrict ourselves to graphs accessible from root, otherwise we may
    # accidentally destroy nodes from other graphs that are users of the
    # constants we use.
    cov = accessible_graphs(root)
    live = dfs(root.return_, True)
    total = _dfs(root.return_, succ_bidirectional(cov))
    dead = set(total) - set(live)
    for node in dead:
        node.inputs.clear()  # type: ignore


###############
# Isomorphism #
###############


def _same_node_shallow(n1, n2, equiv):
    # Works for Constant, Parameter and nodes previously seen
    if n1 in equiv and equiv[n1] is n2:
        return True
    elif is_constant_graph(n1) and is_constant_graph(n2):
        # Note: we provide current equiv so that nested graphs can properly
        # match their free variables, using the equiv of their parent graph.
        return isomorphic(n1.value, n2.value, equiv)
    elif is_constant(n1):
        return n1.value == n2.value
    elif is_parameter(n1):
        # Parameters are matched together in equiv when we ask whether two
        # graphs are isomorphic. Therefore, we only end up here when trying to
        # match free variables.
        return False
    else:
        raise TypeError(n1)  # pragma: no cover


def _same_node(n1, n2, equiv):
    # Works for Apply (when not seen previously) or other nodes
    if is_apply(n1):
        return all(_same_node_shallow(i1, i2, equiv)
                   for i1, i2 in zip(n1.inputs, n2.inputs))
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
    """Return whether g1 and g2 are structurally equivalent.

    Constants are isomorphic iff they contain the same value or are isomorphic
    graphs.

    g1.return_ and g2.return_ must represent the same node under the
    isomorphism. Parameters must match in the same order.
    """
    if len(g1.parameters) != len(g2.parameters):
        return False

    prev_equiv = equiv
    equiv = dict(zip(g1.parameters, g2.parameters))
    if prev_equiv:
        equiv.update(prev_equiv)

    return _same_subgraph(g1.return_, g2.return_, equiv)


##################
# Misc utilities #
##################


def replace(old_node: ANFNode, new_node: ANFNode) -> None:
    """Replace a node by another."""
    uses = set(old_node.uses)
    for node, key in uses:
        node.inputs[key] = new_node


def is_apply(x: ANFNode) -> bool:
    """Return whether x is an Apply."""
    return isinstance(x, Apply)


def is_parameter(x: ANFNode) -> bool:
    """Return whether x is a Parameter."""
    return isinstance(x, Parameter)


def is_constant(x: ANFNode, cls: Any = object) -> bool:
    """Return whether x is a Constant, with value of given cls."""
    return isinstance(x, Constant) and isinstance(x.value, cls)


def is_constant_graph(x: ANFNode) -> bool:
    """Return whether x is a Constant with a Graph value."""
    return is_constant(x, Graph)
