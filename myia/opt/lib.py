"""Library of optimizations."""

from ..graph_utils import dfs
from ..ir import succ_incoming, freevars_boundary, \
    Graph, Constant, is_constant, is_constant_graph, is_apply, \
    GraphCloner
from ..unify import Var, var, SVar
from ..prim import ops as P
from ..utils import Namespace

from .opt import \
    sexp_to_node, \
    PatternSubstitutionOptimization as psub, \
    pattern_replacer


#####################
# Generic variables #
#####################


X = Var('X')
Y = Var('Y')
Z = Var('Z')
X1 = Var('X1')
Y1 = Var('Y1')
X2 = Var('X2')
Y2 = Var('Y2')

C = var(is_constant)
C1 = var(is_constant)
C2 = var(is_constant)
CNS = var(lambda x: is_constant(x, Namespace))
G = var(is_constant_graph)
NIL = var(lambda x: is_constant(x) and x.value == ())

Xs = SVar(Var())
Ys = SVar(Var())
Cs = SVar(var(is_constant))


def primset_var(*prims):
    """Create a variable that matches a Primitive node."""
    return var(lambda node: is_constant(node) and node.value in prims)


###############################
# Tuple-related optimizations #
###############################


@pattern_replacer(P.getitem, (P.cons_tuple, X, Y), C)
def getitem_tuple(node, equiv):
    """Match a constant index in an explicit tuple.

    (a, b, c, ...)[0] => a
    (a, b, c, ...)[1] => b
    ...
    """
    i = equiv[C].value
    assert isinstance(i, int)
    if i == 0:
        return equiv[X]
    else:
        return sexp_to_node((P.getitem, equiv[Y], i - 1), node.graph)


@pattern_replacer(P.setitem, (P.cons_tuple, X, Y), C, Z)
def setitem_tuple(node, equiv):
    """Match a constant setitem in an explicit tuple.

    setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
    setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
    ...
    """
    i = equiv[C].value
    assert isinstance(i, int)
    if i == 0:
        return sexp_to_node((P.cons_tuple, equiv[Z], equiv[Y]), node.graph)
    else:
        return sexp_to_node((P.cons_tuple, equiv[X],
                             (P.setitem, equiv[Y], i - 1, equiv[Z])),
                            node.graph)


# head((a, b, ...)) => a
head_tuple = psub(
    pattern=(P.head, (P.cons_tuple, X, Y)),
    replacement=X,
    name='head_tuple'
)


# tail((a, b, ...)) => (b, ...)
tail_tuple = psub(
    pattern=(P.tail, (P.cons_tuple, X, Y)),
    replacement=Y,
    name='tail_tuple'
)


# f((a, b, ...), (p, q, ...)) => (f(a, p), f(b, q), ...)
# For f in the following list:
_BubbleBinary = primset_var(P.add)

bubble_op_cons_binary = psub(
    pattern=(_BubbleBinary, (P.cons_tuple, X1, Y1), (P.cons_tuple, X2, Y2)),
    replacement=(P.cons_tuple,
                 (_BubbleBinary, X1, X2),
                 (_BubbleBinary, Y1, Y2)),
    name='bubble_op_cons_binary'
)


# f((), ()) => () -- this is a kind of constant prop
bubble_op_nil_binary = psub(
    pattern=(_BubbleBinary, NIL, NIL),
    replacement=NIL,
    name='bubble_op_nil_binary'
)


##############################
# Arithmetic simplifications #
##############################


multiply_by_zero_l = psub(
    pattern=(P.mul, 0, X),
    replacement=0,
    name='multiply_by_zero_l'
)

multiply_by_zero_r = psub(
    pattern=(P.mul, X, 0),
    replacement=0,
    name='multiply_by_zero_r'
)

multiply_by_one_l = psub(
    pattern=(P.mul, 1, X),
    replacement=X,
    name='multiply_by_one_l'
)

multiply_by_one_r = psub(
    pattern=(P.mul, X, 1),
    replacement=X,
    name='multiply_by_one_r'
)

add_zero_l = psub(
    pattern=(P.add, 0, X),
    replacement=X,
    name='add_zero_l'
)

add_zero_r = psub(
    pattern=(P.add, X, 0),
    replacement=X,
    name='add_zero_r'
)


######################
# Branch elimination #
######################


simplify_always_true = psub(
    pattern=(P.if_, True, X, Y),
    replacement=(X,),
    name='simplify_always_true'
)


simplify_always_false = psub(
    pattern=(P.if_, False, X, Y),
    replacement=(Y,),
    name='simplify_always_false'
)


###################
# Resolve globals #
###################


def make_resolver(convert):
    """Create an optimization to resolve globals.

    Args:
        convert: The function to use for conversion.
    """
    @pattern_replacer(P.resolve, CNS, C)
    def resolve_globals(node, equiv):
        ns = equiv[CNS]
        x = equiv[C]
        g = convert(ns.value[x.value])
        return Constant(g)

    return resolve_globals


############
# Inlining #
############


def make_inliner(inline_criterion, check_recursive):
    """Create an inliner.

    Args:
        inline_criterion: A function that takes (graph, node, args) and
            returns whether the graph should be inlined or not.
        check_recursive: Check whether a function is possibly recursive
            before inlining it. If it is, don't inline.
    """
    @pattern_replacer(G, Xs)
    def inline(node, equiv):
        g = equiv[G].value
        args = equiv[Xs]

        if inline_criterion is not None:
            if not inline_criterion(g, node, args):
                return node

        if check_recursive:
            if g.recursive:
                return node

        clone = GraphCloner(total=False)
        clone.add_clone(g, node.graph, args)
        return clone[g.output]

    return inline


def is_trivial_graph(g, node, args):
    """Inline trivial graphs.

    A trivial graph is a graph that contains at most one function
    application.
    """
    nodes = [node
             for node in dfs(g.output,
                             succ_incoming,
                             freevars_boundary(g, False))
             if is_apply(node)]

    if len(nodes) == 0:
        return True
    elif len(nodes) == 1:
        app, = nodes
        return all(not is_constant_graph(inp) for inp in app.inputs[1:])
    else:
        return False


def is_unique_use(g, node, args):
    """Inline graphs that are only used once."""
    users = g.graph_users
    return len(users) == 1 and sum(users.values()) == 1


inline_trivial = make_inliner(inline_criterion=is_trivial_graph,
                              check_recursive=False)

inline_unique_uses = make_inliner(inline_criterion=is_unique_use,
                                  check_recursive=True)

inline = make_inliner(inline_criterion=None, check_recursive=True)


##########################
# Drop calls into graphs #
##########################


@pattern_replacer((G, Xs), Ys)
def drop_into_call(node, equiv):
    """Drop a call into the graph that returns the function.

    g(x)(y) => g2(x, y)

    Where g2 is a modified copy of g that incorporates the call on y.
    """
    g = equiv[G].value
    g2 = GraphCloner(g)[g]

    xs = equiv[Xs]
    ys = equiv[Ys]

    new_output = (g2.output, *ys)

    g2.output = Constant('DUMMY')
    g2.output = sexp_to_node(new_output, g2)

    return sexp_to_node((g2, *xs), node.graph)


@pattern_replacer((P.if_, X, Y, Z), Xs)
def drop_into_if(node, equiv):
    """Drop a call on the result of if into both branches.

    f(if(x, y, z)) => if(x, () -> f(y()), () -> f(z()))
    """
    y = equiv[Y]
    z = equiv[Z]

    y2 = Graph()
    y2.output = sexp_to_node(((y,), *equiv[Xs]), y2)

    z2 = Graph()
    z2.output = sexp_to_node(((z,), *equiv[Xs]), z2)

    new = (P.if_, equiv[X], y2, z2)
    return sexp_to_node(new, node.graph)
