"""Graph optimization routines."""

from weakref import WeakKeyDictionary
from collections import deque

from ..ir import ANFNode, Apply, Constant, Graph, Special, manage
from ..prim import Primitive
from ..utils.unify import Unification, Var


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
                 condition=None,
                 name=None,
                 multigraph=True,
                 interest=False):
        """Initialize va PatternSubstitutionOptimization."""
        g: Var = Var('RootG')
        self.pattern = sexp_to_node(pattern, g, multigraph)
        if callable(replacement):
            self.replacement = replacement
        else:
            self.replacement = sexp_to_node(replacement, g)
        self.unif = Unification()
        self.condition = condition
        self.name = name
        if interest is False:
            if (self.pattern.is_apply() and
                    self.pattern.inputs[0].is_constant(Primitive)):
                interest = self.pattern.inputs[0].value
            else:
                # Maybe warn in this case?
                interest = None
        self.interest = interest

    def __call__(self, optimizer, node):
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
        if equiv is not None:
            if callable(self.replacement):
                return self.replacement(optimizer, node, equiv)
            elif self.condition is None or self.condition(equiv):
                return self.unif.reify(self.replacement, equiv)
        else:
            return None


def pattern_replacer(*pattern):
    """Create a PatternSubstitutionOptimization using this function."""
    if len(pattern) == 2 and pattern[0] == 'just':
        pattern = pattern[1]

    def deco(f):
        return PatternSubstitutionOptimization(pattern, f, name=f.__name__)
    return deco


class NodeMap:
    """Mapping of node to optimizer.

    This helps global optimizers select the relevant optimizers to
    apply for each node.

    Optimizers that are mapped to None are considered relevant for all
    nodes.

    Other than None, only primitives are currently supported as interests.

    """

    def __init__(self):
        """Create a NodeMap."""
        self._d = dict()

    def register(self, interests, opt=None):
        """Register an optimizer for some interests."""
        def do_register(opt):
            ints = interests
            if ints is None:
                self._d.setdefault(None, []).append(opt)
                return
            if not isinstance(ints, tuple):
                ints = (ints,)
            for interest in ints:
                assert isinstance(interest, Primitive)
                self._d.setdefault(interest, []).append(opt)

        # There could be the option to return do_register also.
        do_register(opt)

    def get(self, node):
        """Get a list of optimizers that could apply for a node."""
        res = []
        res.extend(self._d.get(None, []))
        if node.is_apply() and node.inputs[0].is_constant():
            res.extend(self._d.get(node.inputs[0].value, []))
        return res


class LocalPassOptimizer:
    """Apply a set of local optimizations in bfs order."""

    def __init__(self, node_map, optimizer=None):
        """Initialize a LocalPassOptimizer."""
        self.node_map = node_map
        self.optimizer = optimizer

    def __call__(self, graph):
        """Apply optimizations on given graphs in node order.

        This will visit the nodes from the output to the inputs in a
        bfs manner while avoiding parts of the graph that are dropped
        due to optimizations.
        """
        if self.optimizer is not None:
            mng = self.optimizer.resources.manager
            mng.add_graph(graph)
        else:
            mng = manage(graph)

        seen = set([graph])
        todo = deque()
        changes = False
        todo.append(graph.output)

        while len(todo) > 0:
            n = todo.popleft()
            if n in seen or n not in mng.all_nodes:
                continue
            seen.add(n)

            new, chg = self.apply_opt(mng, n)

            changes |= chg

            if new.is_constant(Graph):
                if new.value not in seen:
                    todo.appendleft(new.value.output)
                    seen.add(new.value)
            else:
                todo.extendleft(reversed(new.inputs))

            if chg:
                # If changes, re-do the parent node(s)
                uses = set(u[0] for u in mng.uses[new])
                # TODO: grab all constants for a graph
                seen.difference_update(uses)
                todo.extendleft(uses)

        return changes

    def apply_opt(self, mng, n):
        """Apply optimizations passes according to the node map."""
        loop = True
        changes = False
        while loop:
            loop = False
            for transformer in self.node_map.get(n):
                new = transformer(self.optimizer, n)
                if new is True:
                    changes = True
                    continue
                if new and new is not n:
                    new.expect_inferred.update(n.inferred)
                    mng.replace(n, new)
                    n = new
                    loop = True
                    changes = True
                    break

        return n, changes


class GraphTransform:
    """Represents a graph transform.

    The transform of a graph is unique and it is stored in graph.transforms.
    Here are examples of graph transforms:

    * A graph's gradient.
    * A copy of the graph, except the output is called.
    * A copy of the graph, except it returns the ith element of the output.
    """

    def __init__(self, compute):
        """Initialize a GraphTransform."""
        self.cache = WeakKeyDictionary()
        self.compute = compute

    def __call__(self, graph, *args):
        """Return the transformed graph.

        Computes the transform if it isn't already available.
        """
        if graph not in self.cache:
            self.cache[graph] = {}
        cache = self.cache[graph]
        if args not in cache:
            cache[args] = self.compute(graph, *args)
        return cache[args]
