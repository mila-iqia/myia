"""Library of optimizations."""

from ..ir import Graph, Constant, GraphCloner
from ..prim import Primitive, ops as P
from ..utils import Namespace
from ..utils.unify import Var, var, SVar

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


def _is_c(n):
    return n.is_constant()


def _is_cg(n):
    return n.is_constant_graph()


C = var(_is_c)
C1 = var(_is_c)
C2 = var(_is_c)
CNS = var(lambda x: x.is_constant(Namespace))
G = var(_is_cg)
NIL = var(lambda x: x.is_constant() and x.value == ())

Xs = SVar(Var())
Ys = SVar(Var())
Cs = SVar(var(_is_c))


def primset_var(*prims):
    """Create a variable that matches a Primitive node."""
    return var(lambda node: node.is_constant() and node.value in prims)


###############################
# Tuple-related optimizations #
###############################


@pattern_replacer(P.tuple_getitem, (P.make_tuple, Xs), C)
def getitem_tuple(optimizer, node, equiv):
    """Match a constant index in an explicit tuple.

    (a, b, c, ...)[0] => a
    (a, b, c, ...)[1] => b
    ...
    """
    i = equiv[C].value
    assert isinstance(i, int)
    return equiv[Xs][i]


@pattern_replacer(P.tuple_setitem, (P.make_tuple, Xs), C, Z)
def setitem_tuple(optimizer, node, equiv):
    """Match a constant setitem in an explicit tuple.

    setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
    setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
    ...
    """
    i = equiv[C].value
    assert isinstance(i, int)
    elems = list(equiv[Xs])
    elems[i] = equiv[Z]
    return sexp_to_node((P.make_tuple, *elems), node.graph)


# tail((a, b, ...)) => (b, ...)
tail_tuple = psub(
    pattern=(P.tail, (P.make_tuple, X, Xs)),
    replacement=(P.make_tuple, Xs),
    name='tail_tuple'
)


# f((a, b, ...), (p, q, ...)) => (f(a, p), f(b, q), ...)
# For f in the following list:
_BubbleBinary = primset_var(P.scalar_add)


@pattern_replacer(_BubbleBinary, (P.make_tuple, Xs), (P.make_tuple, Ys))
def bubble_op_tuple_binary(optimizer, node, equiv):
    """Replace F((x, y, ...), (a, b, ...)) => (F(x, a), F(y, b), ...).

    Only works for a specific list of Fs.
    """
    xs = equiv[Xs]
    ys = equiv[Ys]
    op = equiv[_BubbleBinary]
    assert len(xs) == len(ys)
    elems = [(op, x, y) for x, y in zip(xs, ys)]
    return sexp_to_node((P.make_tuple, *elems), node.graph)


##############################
# Arithmetic simplifications #
##############################


multiply_by_zero_l = psub(
    pattern=(P.scalar_mul, 0, X),
    replacement=0,
    name='multiply_by_zero_l'
)

multiply_by_zero_r = psub(
    pattern=(P.scalar_mul, X, 0),
    replacement=0,
    name='multiply_by_zero_r'
)

multiply_by_one_l = psub(
    pattern=(P.scalar_mul, 1, X),
    replacement=X,
    name='multiply_by_one_l'
)

multiply_by_one_r = psub(
    pattern=(P.scalar_mul, X, 1),
    replacement=X,
    name='multiply_by_one_r'
)

add_zero_l = psub(
    pattern=(P.scalar_add, 0, X),
    replacement=X,
    name='add_zero_l'
)

add_zero_r = psub(
    pattern=(P.scalar_add, X, 0),
    replacement=X,
    name='add_zero_r'
)

elim_identity = psub(
    pattern=(P.identity, X),
    replacement=X,
    name='elim_identity'
)


######################
# Branch elimination #
######################


simplify_always_true = psub(
    pattern=(P.switch, True, X, Y),
    replacement=X,
    name='simplify_always_true'
)


simplify_always_false = psub(
    pattern=(P.switch, False, X, Y),
    replacement=Y,
    name='simplify_always_false'
)

#####################
# Simplify partials #
#####################


simplify_partial = psub(
    pattern=((P.partial, X, Xs), Ys),
    replacement=(X, Xs, Ys),
    name='simplify_partial'
)


###################
# Resolve globals #
###################


@pattern_replacer(P.resolve, CNS, C)
def resolve_globals(optimizer, node, equiv):
    """Resolve global variables."""
    ns = equiv[CNS]
    x = equiv[C]
    res = optimizer.resources.convert(ns.value[x.value])
    return Constant(res)


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
    def inline(optimizer, node, equiv):
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
    nodes = [node for node in g.nodes if node.is_apply()]
    # One node will be the return node, so len(nodes) == 1 if the node
    # returns a constant or a free variable.
    return len(nodes) <= 2


def is_unique_use(g, node, args):
    """Inline graphs that are only used once."""
    users = g.graph_users
    return len(users) == 1 and sum(users.values()) == 1


def is_core(g, node, args):
    """Inline graphs that are marked as part of the core."""
    return g.flags.get('core', False)


inline_trivial = make_inliner(inline_criterion=is_trivial_graph,
                              check_recursive=False)

inline_unique_uses = make_inliner(inline_criterion=is_unique_use,
                                  check_recursive=True)

inline_core = make_inliner(inline_criterion=is_core,
                           check_recursive=False)

inline = make_inliner(inline_criterion=None, check_recursive=True)


@pattern_replacer('just', G)
def replace_applicator(optimizer, node, equiv):
    """Replace a function that applies another by the other function.

    For example, `lambda x, y: f(x, y)` is replaced by f.

    The inner function must be applied on all the outer function's parameters
    in the exact same order, and it must be either a Primitive or a global
    function.
    """
    g = equiv[G].value
    out = g.output
    if out.is_apply() and out.inputs[1:] == g.parameters:
        inner = out.inputs[0]
        # NOTE: it is likely correct to use `inner.value.parent is not g` as
        # the condition instead of `is None`, the current code is just playing
        # it safe.
        if inner.is_constant(Primitive) \
                or inner.is_constant_graph() and inner.value.parent is None:
            return inner
    return node


##########################
# Drop calls into graphs #
##########################


@pattern_replacer((G, Xs), Ys)
def drop_into_call(optimizer, node, equiv):
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


@pattern_replacer(((P.switch, X, Y, Z),), Xs)
def drop_into_if(optimizer, node, equiv):
    """Drop a call on the result of if into both branches.

    f(if(x, y, z)) => if(x, () -> f(y()), () -> f(z()))
    """
    y = equiv[Y]
    z = equiv[Z]

    y2 = Graph()
    y2.output = sexp_to_node(((y,), *equiv[Xs]), y2)

    z2 = Graph()
    z2.output = sexp_to_node(((z,), *equiv[Xs]), z2)

    new = ((P.switch, equiv[X], y2, z2),)
    return sexp_to_node(new, node.graph)


#################
# Gradient opts #
#################


# J(Jinv(x)) ==> x
elim_j_jinv = psub(
    pattern=(P.J, (P.Jinv, X)),
    replacement=X,
    name='elim_j_jinv'
)


# Jinv(J(x)) ==> x
elim_jinv_j = psub(
    pattern=(P.Jinv, (P.J, X)),
    replacement=X,
    name='elim_jinv_j'
)


@pattern_replacer(P.J, C)
def expand_J(optimizer, node, equiv):
    """Replaces a call to J(f) by the graph for J(f).

    This will not replace J(x) when x is not a constant graph.
    """
    from ..grad import J as Jimpl
    arg = equiv[C].value
    try:
        newg = Jimpl(arg, optimizer.resources.manager)
    except NotImplementedError:
        return None
    return Constant(newg)
