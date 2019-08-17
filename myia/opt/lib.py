"""Library of optimizations."""

from .. import operations
from ..abstract import (
    ANYTHING,
    DEAD,
    SHAPE,
    AbstractArray,
    AbstractFunction,
    AbstractJTagged,
    abstract_clone,
    type_token,
)
from ..composite import gadd, zeros_like
from ..dtype import Number
from ..ir import (
    Apply,
    BasicRemapper,
    Constant,
    Graph,
    GraphCloner,
    transformable_clone,
)
from ..prim import Primitive, ops as P
from ..utils import Namespace, Partializable, overload
from ..utils.unify import SVar, Var, var
from .opt import (
    GraphTransform,
    PatternSubstitutionOptimization as psub,
    pattern_replacer,
    sexp_to_node,
)

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
X3 = Var('X3')
X4 = Var('X4')
X5 = Var('X5')


def _is_c(n):
    return n.is_constant()


def _is_cg(n):
    return n.is_constant_graph()


C = var(_is_c)
C1 = var(_is_c)
C2 = var(_is_c)
CNS = var(lambda x: x.is_constant(Namespace))
G = var(_is_cg)
G1 = var(_is_cg)
G2 = var(_is_cg)
NIL = var(lambda x: x.is_constant() and x.value == ())

Xs = SVar(Var())
Ys = SVar(Var())
Cs = SVar(var(_is_c))


def M(mg):
    """Create a variable that matches a Metagraph."""
    def chk(x):
        return (x.is_constant_graph()
                and x.value.flags.get('metagraph') == mg)
    return var(chk)


def primset_var(*prims):
    """Create a variable that matches a Primitive node."""
    return var(lambda node: node.is_constant() and node.value in prims)


AA = AbstractArray(ANYTHING, {SHAPE: ANYTHING})


###############################
# Tuple-related optimizations #
###############################


@pattern_replacer(P.tuple_getitem, (P.tuple_setitem, X, C1, Y), C2)
def getitem_setitem_tuple(optimizer, node, equiv):
    """Simplify getitem over setitem.

    setitem(xs, 0, v)[0] => v
    setitem(xs, 0, v)[1] => xs[1]
    """
    i1 = equiv[C1].value
    i2 = equiv[C2].value
    if i1 == i2:
        return equiv[Y]
    else:
        return node.graph.apply(P.tuple_getitem, equiv[X], i2)


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


@pattern_replacer(P.tuple_setitem, C1, C2, Z)
def setitem_tuple_ct(optimizer, node, equiv):
    """Match a constant setitem in an explicit tuple.

    setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
    setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
    ...
    """
    tup = equiv[C1].value
    i = equiv[C2].value
    assert isinstance(tup, tuple)
    assert isinstance(i, int)
    elems = list(tup)
    elems[i] = equiv[Z]
    return sexp_to_node((P.make_tuple, *elems), node.graph)


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


###########################
# gadd optimizations #
###########################


_gadd = M(gadd)
_zlk = M(zeros_like)


gadd_zero_l = psub(
    pattern=(_gadd, (_zlk, X), Y),
    replacement=Y,
    name='gadd_zero_l'
)


gadd_zero_r = psub(
    pattern=(_gadd, Y, (_zlk, X)),
    replacement=Y,
    name='gadd_zero_r'
)


gadd_switch = psub(
    pattern=(_gadd,
             (P.switch, Y, X1, X2),
             (P.switch, Y, X3, X4)),
    replacement=(P.switch, Y,
                 (gadd, X1, X3),
                 (gadd, X2, X4)),
    name='gadd_switch'
)


##############################
# Arithmetic simplifications #
##############################


_ArrayType = Var('ArrayType')


@overload
def _transform(pattern: tuple):
    f, *args = pattern
    return (P.array_map, f, *tuple(_transform(arg) for arg in args))


@overload  # noqa: F811
def _transform(pattern: Var):
    return pattern


@overload  # noqa: F811
def _transform(pattern: (int, float)):
    return (P.distribute, (P.scalar_to_array, pattern, _ArrayType),
            var(_is_c))


def on_array_map(orig):
    """Create an optimization on array_map from a scalar optimization.

    Original pattern: (f, x, y)
    New pattern:      (array_map, f, x, y)

    Original pattern: (f, x, 1)
    New pattern:      (array_map, f, x, distribute(scalar_to_array(1)))

    Etc.
    """
    return psub(
        pattern=_transform(orig.sexp),
        replacement=_transform(orig.sexp_replacement),
        name=f'{orig.name}_map',
    )


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

usub_cancel = psub(
    pattern=(P.scalar_usub, (P.scalar_usub, X)),
    replacement=X,
    name='usub_cancel'
)

usub_sink_mul_l = psub(
    pattern=(P.scalar_mul, (P.scalar_usub, X), Y),
    replacement=(P.scalar_usub, (P.scalar_mul, X, Y)),
    name='usub_sink_mul_l'
)

usub_sink_mul_r = psub(
    pattern=(P.scalar_mul, X, (P.scalar_usub, Y)),
    replacement=(P.scalar_usub, (P.scalar_mul, X, Y)),
    name='usub_sink_mul_r'
)

usub_sink_div_l = psub(
    pattern=(P.scalar_div, (P.scalar_usub, X), Y),
    replacement=(P.scalar_usub, (P.scalar_div, X, Y)),
    name='usub_sink_div_l'
)

usub_sink_div_r = psub(
    pattern=(P.scalar_div, X, (P.scalar_usub, Y)),
    replacement=(P.scalar_usub, (P.scalar_div, X, Y)),
    name='usub_sink_div_r'
)

divdiv_to_mul = psub(
    pattern=(P.scalar_div, (P.scalar_div, X, Y), Z),
    replacement=(P.scalar_div, X, (P.scalar_mul, Y, Z)),
    name='divdiv_to_mul'
)

add_usub = psub(
    pattern=(P.scalar_add, X, (P.scalar_usub, Y)),
    replacement=(P.scalar_sub, X, Y),
    name='add_usub'
)

sub_usub = psub(
    pattern=(P.scalar_sub, X, (P.scalar_usub, Y)),
    replacement=(P.scalar_add, X, Y),
    name='sub_usub'
)

elim_identity = psub(
    pattern=(P.identity, X),
    replacement=X,
    name='elim_identity'
)

multiply_by_zero_l_map = on_array_map(multiply_by_zero_l)
multiply_by_zero_r_map = on_array_map(multiply_by_zero_r)
multiply_by_one_l_map = on_array_map(multiply_by_one_l)
multiply_by_one_r_map = on_array_map(multiply_by_one_r)
add_zero_l_map = on_array_map(add_zero_l)
add_zero_r_map = on_array_map(add_zero_r)
usub_cancel_map = on_array_map(usub_cancel)
usub_sink_mul_l_map = on_array_map(usub_sink_mul_l)
usub_sink_mul_r_map = on_array_map(usub_sink_mul_r)
usub_sink_div_l_map = on_array_map(usub_sink_div_l)
usub_sink_div_r_map = on_array_map(usub_sink_div_r)
divdiv_to_mul_map = on_array_map(divdiv_to_mul)
add_usub_map = on_array_map(add_usub)
sub_usub_map = on_array_map(sub_usub)


#########################
# Array simplifications #
#########################


# distribute(x, shp) => x when x.shape == shp
elim_distribute = psub(
    pattern=(P.distribute, X, C),
    replacement=X,
    condition=lambda equiv: equiv[X].shape == equiv[C].value,
    name='elim_distribute'
)


# array_reduce(f, x, shp) => x when x.shape == shp
elim_array_reduce = psub(
    pattern=(P.array_reduce, Y, X, C),
    replacement=X,
    condition=lambda equiv: equiv[X].shape == equiv[C].value,
    name='elim_array_reduce'
)


@pattern_replacer(P.transpose, X, C)
def elim_transpose(optimizer, node, equiv):
    """Remove transposes that correspond to identity."""
    axes = equiv[C].value
    if axes == tuple(range(len(axes))):
        return equiv[X]
    else:
        return node


@pattern_replacer(P.transpose, (P.transpose, X, C1), C2)
def merge_transposes(optimizer, node, equiv):
    """Merge transpose operations into a single transpose."""
    axes1 = equiv[C1].value
    axes2 = equiv[C2].value
    assert len(axes1) == len(axes2)
    axes_final = tuple(axes1.index(x) for x in axes2)
    return node.graph.apply(P.transpose, equiv[X], axes_final)


@pattern_replacer(P.array_map, G, Xs)
def unfuse_composite(optimizer, node, equiv):
    """Tranform array_map on a graph to a graph of array_maps.

    This must be applied to scalar-only graphs.
    """
    # This has to be defined inline because of circular imports
    class UnfuseRemapper(BasicRemapper):
        def __init__(self, g, shape):
            super().__init__(
                graphs=g.graphs_used.keys() | {g},
                relation='unfused'
            )
            self.shape = shape

        def asarray(self, ng, i):
            if i.is_constant():
                return ng.apply(P.distribute, ng.apply(P.scalar_to_array, i,
                                                       AA),
                                self.shape)
            else:
                return i

        def link_apply(self, link):
            ng = link.new_graph
            node = link.node
            assert node.inputs[0].is_constant(Primitive)
            ni = [self.asarray(ng, self.repl[i]) for i in node.inputs[1:]]
            link.new_node.inputs = \
                [ng.constant(P.array_map), node.inputs[0]] + ni

        def finalize_graph(self, g, ng):
            # This fails if we set .return_ instead of .output, not sure why.
            ng.output = self.repl[g.output]

    g = equiv[G].value
    xs = equiv[Xs]
    r = UnfuseRemapper(g, xs[0].shape)
    r.run()
    ng = r.get_graph(g)
    return node.graph.apply(ng, *xs)


@pattern_replacer(P.array_map, G, Xs)
def simplify_array_map(optimizer, node, equiv):
    """Simplify array_map on certain graphs.

    If the graph cannot be eliminated, it is marked with the flag
    `inline_inside`, meaning that all calls within it must be inlined.

    Examples:
        array_map(lambda x, y: f(x, y), xs, ys)
            => array_map(f, xs, ys)

        array_map(lambda x, y: f(y, x), xs, ys)
            => array_map(f, ys, xs)

        array_map(lambda x, y: x, xs, ys)
            => xs

        array_map(lambda x: f(x, 3), xs)
            => array_map(f, xs, distribute(scalar_to_array(3), shape(xs)))

    """
    g = equiv[G].value
    xs = equiv[Xs]

    def to_outer(x):
        if x.is_parameter():
            idx = g.parameters.index(x)
            return xs[idx]
        elif x.is_constant() \
                and issubclass(type_token(x.abstract), Number):
            shp = (P.shape, xs[0])
            sexp = (P.distribute, (P.scalar_to_array, x, AA), shp)
            return sexp_to_node(sexp, node.graph)
        else:
            # Raise a semi-rare exception that won't hide bugs
            raise NotImplementedError()

    if len(g.scope) > 1:
        return node

    out = g.output

    try:
        if out.is_parameter() or out.is_constant():
            return to_outer(out)
        elif out.inputs[0].is_constant():
            args = [to_outer(arg) for arg in out.inputs[1:]]
            return node.graph.apply(P.array_map, out.inputs[0], *args)
        else:
            return node  # pragma: no cover

    except NotImplementedError:
        if g.has_flags('inline_inside'):
            return node
        else:
            g.set_flags('inline_inside')
            return True


#############################
# Env-related optimizations #
#############################


@pattern_replacer(P.env_getitem, (P.env_setitem, X, C1, Y), C2, Z)
def cancel_env_set_get(optimizer, node, equiv):
    """Simplify combinations of env_get/setitem.

    * get(set(env, k1, v), k2, dflt) =>
        * v                  when k1 == k2
        * get(env, k2, dflt) when k1 != k2
    """
    key1 = equiv[C1]
    key2 = equiv[C2]
    if key1.value == key2.value:
        return equiv[Y]
    else:
        sexp = (P.env_getitem, equiv[X], key2, equiv[Z])
        return sexp_to_node(sexp, node.graph)


# getitem(newenv, key, default) => default
getitem_newenv = psub(
    pattern=(P.env_getitem, C1, C2, Y),
    replacement=Y,
    condition=lambda equiv: len(equiv[C1].value) == 0,
    name='getitem_newenv'
)


# getitem(env_add(e1, e2), key, default)
#     => gadd(getitem(e1, key, default), getitem(e2, key, default))
getitem_env_add = psub(
    pattern=(P.env_getitem, (P.env_add, X, Y), C, Z),
    replacement=(gadd,
                 (P.env_getitem, X, C, Z),
                 (P.env_getitem, Y, C, Z)),
    name='getitem_env_add'
)


# setitem(e, key, DEAD) => e
setitem_dead = psub(
    pattern=(P.env_setitem, X, Y, DEAD),
    replacement=X,
    name='setitem_dead'
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


# Simplify nested switch with the same condition (case 1)
simplify_switch1 = psub(
    pattern=(P.switch, X1, (P.switch, X1, X2, X3), X4),
    replacement=(P.switch, X1, X2, X4),
    name='simplify_switch1'
)


# Simplify nested switch with the same condition (case 2)
simplify_switch2 = psub(
    pattern=(P.switch, X1, X2, (P.switch, X1, X3, X4)),
    replacement=(P.switch, X1, X2, X4),
    name='simplify_switch2'
)


# Simplify switch when both branches are the same node
simplify_switch_idem = psub(
    pattern=(P.switch, X, Y, Y),
    replacement=Y,
    name='simplify_switch_idem'
)


_PutInSwitch_l = (
    P.scalar_add,
    P.scalar_sub,
    P.scalar_mul,
    P.scalar_div,
    P.scalar_mod,
    P.scalar_pow,
)

_PutInSwitch = primset_var(*_PutInSwitch_l)

# Binary operations on switches with same conditions are transformed into
# a switch on two operations, e.g.
# switch(x, a, b) + switch(x, c, d) => switch(x, a + c, b + d)
combine_switches = psub(
    pattern=(_PutInSwitch, (P.switch, X1, X2, X3), (P.switch, X1, X4, X5)),
    replacement=(P.switch, X1, (_PutInSwitch, X2, X4), (_PutInSwitch, X3, X5)),
    name='combine_switches',
    interest=_PutInSwitch_l
)


combine_switches_array = psub(
    pattern=(P.array_map,
             _PutInSwitch,
             (P.switch, X1, X2, X3),
             (P.switch, X1, X4, X5)),
    replacement=(P.switch, X1,
                 (P.array_map, _PutInSwitch, X2, X4),
                 (P.array_map, _PutInSwitch, X3, X5)),
    name='combine_switches_array'
)


float_tuple_getitem_through_switch = psub(
    pattern=(P.tuple_getitem, (P.switch, X1, X2, X3), C),
    replacement=(P.switch, X1,
                 (P.tuple_getitem, X2, C),
                 (P.tuple_getitem, X3, C)),
    name='float_tuple_getitem_through_switch'
)


float_env_getitem_through_switch = psub(
    pattern=(P.env_getitem, (P.switch, X1, X2, X3), X4, X5),
    replacement=(P.switch, X1,
                 (P.env_getitem, X2, X4, X5),
                 (P.env_getitem, X3, X4, X5)),
    name='float_env_getitem_through_switch'
)


#####################
# Simplify partials #
#####################


simplify_partial = psub(
    pattern=((P.partial, X, Xs), Ys),
    replacement=(X, Xs, Ys),
    name='simplify_partial',
    interest=Apply,
)


###################
# Resolve globals #
###################


@pattern_replacer(operations.resolve, CNS, C)
def resolve_globals(optimizer, node, equiv):
    """Resolve global variables."""
    ns = equiv[CNS]
    x = equiv[C]
    res = optimizer.resources.convert(ns.value[x.value])
    return Constant(res)


############
# Inlining #
############


def make_inliner(inline_criterion, check_recursive, name):
    """Create an inliner.

    Args:
        inline_criterion: A function that takes (graph, node, args) and
            returns whether the graph should be inlined or not.
        check_recursive: Check whether a function is possibly recursive
            before inlining it. If it is, don't inline.
        name: The name of the optimization.
    """
    @pattern_replacer(G, Xs, interest=Graph)
    def inline(optimizer, node, equiv):
        g = equiv[G].value
        args = equiv[Xs]

        if inline_criterion is not None:
            if not inline_criterion(g, node, args):
                return node

        if check_recursive:
            if g.recursive:
                return node

        clone = GraphCloner(inline=(g, node.graph, args), total=False)
        return clone[g.output]

    inline.name = name
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
    return g.has_flags('core')


def caller_is_marked(g, node, args):
    """Inline into graphs that are marked."""
    return node.graph.has_flags('inline_inside')


inline_trivial = make_inliner(inline_criterion=is_trivial_graph,
                              check_recursive=True,
                              name='inline_trivial')

inline_unique_uses = make_inliner(inline_criterion=is_unique_use,
                                  check_recursive=True,
                                  name='inline_unique_uses')

inline_core = make_inliner(inline_criterion=is_core,
                           check_recursive=False,
                           name='inline_core')

inline_inside_marked_caller = \
    make_inliner(inline_criterion=caller_is_marked,
                 check_recursive=False,
                 name='inline_inside_marked_caller')

inline = make_inliner(inline_criterion=None,
                      check_recursive=True,
                      name='inline')


@pattern_replacer('just', G, interest=None)
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


##################
# Specialization #
##################


def check_used_once(g):
    """Returns True if a graph has only one usage."""
    mng = g.manager
    return sum(mng.graph_users[g].values()) == 1


@GraphTransform
def specialize_transform(graph, args):
    """Specialize on provided non-None args.

    Parameters that are specialized on are removed.
    """
    mng = graph.manager
    graph = transformable_clone(graph, relation=f'sp')
    mng.add_graph(graph)
    for p, arg in zip(graph.parameters, args):
        if arg is not None:
            mng.replace(p, Constant(arg))
    new_parameters = [p for p, arg in zip(graph.parameters, args)
                      if arg is None]
    mng.set_parameters(graph, new_parameters)
    return graph


@pattern_replacer(G, Xs, interest=Graph)
def specialize_on_graph_arguments(optimizer, node, equiv):
    """Specialize a call on constant graph arguments."""
    g = equiv[G].value
    xs = equiv[Xs]
    specialize = [x.is_constant((Graph, Primitive)) for x in xs]
    if not any(specialize):
        return node
    specialize_map = tuple(x.value if s else None
                           for x, s in zip(xs, specialize))
    new_xs = [x for x, s in zip(xs, specialize) if not s]
    g2 = specialize_transform(g, specialize_map)
    return node.graph.apply(g2, *new_xs)


#################
# Incorporation #
#################


@GraphTransform
def getitem_transform(graph, idx):
    """Map to a graph that only returns the idx-th output.

    If idx == 1, then:

    (x -> (a, b, c)) => (x -> b)
    """
    graph = transformable_clone(graph, relation=f'[{idx}]')
    if graph.output.is_apply(P.make_tuple):
        graph.output = graph.output.inputs[idx + 1]
    else:
        graph.output = graph.apply(P.tuple_getitem, graph.output, idx)
    return graph


@pattern_replacer(P.tuple_getitem, (G, Xs), C)
def incorporate_getitem(optimizer, node, equiv):
    """Incorporate a getitem into a call.

    For example:

        (lambda x: (a, b, c, ...))(x)[0]
            => (lambda x: a)(x)
    """
    g = equiv[G].value
    idx = equiv[C].value
    if check_used_once(g):
        return node.graph.apply(getitem_transform(g, idx), *equiv[Xs])


@pattern_replacer(P.tuple_getitem, ((P.switch, X, G1, G2), Xs), C)
def incorporate_getitem_through_switch(optimizer, node, equiv):
    """Incorporate a getitem into both branches.

    Example:

        switch(x, f, g)(y)[i]
        => switch(x, f2, g2)(y)

    Where f2 and g2 are modified versions of f and g that return their
    ith element.
    """
    g1 = equiv[G1].value
    g2 = equiv[G2].value
    idx = equiv[C].value
    xs = equiv[Xs]

    if check_used_once(g1) and check_used_once(g2):
        g1t = getitem_transform(g1, idx)
        g2t = getitem_transform(g2, idx)

        new = ((P.switch, equiv[X], g1t, g2t), *xs)
        return sexp_to_node(new, node.graph)


@GraphTransform
def env_getitem_transform(graph, key, default):
    """Map to a graph that incorporates a call to env_getitem."""
    rel = getattr(key, 'node', key)
    graph = transformable_clone(graph, relation=f'[{rel}]')
    out = graph.output
    while out.is_apply(P.env_setitem):
        _, out, key2, value = out.inputs
        if key == key2.value:
            graph.output = value
            return graph
    graph.output = graph.apply(P.env_getitem, out, key, default)
    return graph


@pattern_replacer(P.env_getitem, (G, Xs), C, Y)
def incorporate_env_getitem(optimizer, node, equiv):
    """Incorporate an env_getitem into a call."""
    g = equiv[G].value
    key = equiv[C].value
    dflt = equiv[Y]
    if check_used_once(g):
        return node.graph.apply(env_getitem_transform(g, key, dflt),
                                *equiv[Xs])


@pattern_replacer(P.env_getitem, ((P.switch, X, G1, G2), Xs), C, Y)
def incorporate_env_getitem_through_switch(optimizer, node, equiv):
    """Incorporate an env_getitem into both branches."""
    g1 = equiv[G1].value
    g2 = equiv[G2].value
    key = equiv[C].value
    dflt = equiv[Y]
    xs = equiv[Xs]

    if check_used_once(g1) and check_used_once(g2):
        g1t = env_getitem_transform(g1, key, dflt)
        g2t = env_getitem_transform(g2, key, dflt)

        new = ((P.switch, equiv[X], g1t, g2t), *xs)
        return sexp_to_node(new, node.graph)


@GraphTransform
def call_output_transform(graph, nargs):
    """Map to a graph that calls its output.

    ((*args1) -> (*args2) -> f) => (*args1, *args2) -> f(*args2)
    """
    graph = transformable_clone(graph, relation='call')
    newp = [graph.add_parameter() for _ in range(nargs)]
    graph.output = graph.apply(graph.output, *newp)
    return graph


@pattern_replacer((G, Xs), Ys, interest=Apply)
def incorporate_call(optimizer, node, equiv):
    """Incorporate a call into the graph that returns the function.

    Example:

        g(x)(y) => g2(x, y)

    Where g2 is a modified copy of g that incorporates the call on y.
    """
    g = equiv[G].value
    xs = equiv[Xs]
    ys = equiv[Ys]
    if check_used_once(g):
        g2 = call_output_transform(g, len(ys))
        return node.graph.apply(g2, *xs, *ys)


@pattern_replacer(((P.switch, X, G1, G2), Xs), Ys, interest=Apply)
def incorporate_call_through_switch(optimizer, node, equiv):
    """Incorporate a call to both branches.

    Example:

        switch(x, f, g)(y)(z)
        => switch(x, f2, g2)(y, z)

    Where f2 and g2 are modified copies of f and g that incorporate the
    call on both y and z.
    """
    g1 = equiv[G1].value
    g2 = equiv[G2].value
    xs = equiv[Xs]
    ys = equiv[Ys]

    if check_used_once(g1) and check_used_once(g2):
        g1t = call_output_transform(g1, len(ys))
        g2t = call_output_transform(g2, len(ys))

        new = ((P.switch, equiv[X], g1t, g2t), *xs, *ys)
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
        newg = Jimpl(arg, optimizer.resources)
    except NotImplementedError:
        return None
    return Constant(newg)


@abstract_clone.variant
def _jelim_retype(self, j: AbstractJTagged):
    return _jelim_retype_helper(j.element)


@abstract_clone.variant
def _jelim_retype_helper(self, f: AbstractFunction):
    raise TypeError('Function found')


class JElim(Partializable):
    """Eliminate J, iff it is only applied to non-functions."""

    def __init__(self, optimizer):
        """Initialize JElim."""
        self.optimizer = optimizer

    def __call__(self, root):
        """Apply JElim on root."""
        mng = self.optimizer.resources.manager
        mng.keep_roots(root)
        nodes = []
        typesubs = []
        for node in mng.all_nodes:
            try:
                newtype = _jelim_retype(node.abstract)
            except TypeError:
                return False
            if node.is_apply(P.J) or node.is_apply(P.Jinv):
                _, x = node.inputs
                nodes.append((node, x))
            typesubs.append((node, newtype))

        with mng.transact() as tr:
            for node, repl in nodes:
                tr.replace(node, repl)

        for node, newtype in typesubs:
            node.abstract = newtype

        return len(nodes) > 0
