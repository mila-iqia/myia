
import pytest
from myia.api import parse
from myia.ir import Constant, is_constant, isomorphic, GraphCloner
from myia.opt import PatternSubstitutionOptimization as psub, \
    PatternEquilibriumOptimizer, pattern_replacer, sexp_to_graph
from myia.prim import Primitive, ops as prim
from myia.utils.unify import Var, var

X = Var('X')
V = var(is_constant)


# We will optimize patterns of these fake primitives


P = Primitive('P')
Q = Primitive('Q')
R = Primitive('R')


idempotent_P = psub(
    (P, (P, X)),
    (P, X),
    name='idempotent_P'
)

elim_R = psub(
    (R, X),
    X,
    name='elim_R'
)

Q0_to_R = psub(
    (Q, 0),
    (R, 0),
    name='Q0_to_R'
)

QP_to_QR = psub(
    (Q, (P, X)),
    (Q, (R, X)),
    name='QP_to_QR'
)

multiply_by_zero_l = psub(
    (prim.scalar_mul, 0, X),
    0,
    name='multiply_by_zero_l'
)

multiply_by_zero_r = psub(
    (prim.scalar_mul, X, 0),
    0,
    name='multiply_by_zero_r'
)

add_zero_l = psub(
    (prim.scalar_add, 0, X),
    X,
    name='add_zero_l'
)

add_zero_r = psub(
    (prim.scalar_add, X, 0),
    X,
    name='add_zero_r'
)


def _check_opt(before, after, *opts):
    gbefore = parse(before)
    gbefore = GraphCloner(gbefore, total=True)[gbefore]
    gafter = parse(after)

    eq = PatternEquilibriumOptimizer(*opts)
    eq(gbefore)

    assert isomorphic(gbefore, gafter)


def test_checkopt_is_cloning():
    # checkopt should optimize a copy of the before graph, which means
    # _check_opt(before, before) can only succeed if no optimizations are
    # applied.

    def before(x):
        return R(x)

    _check_opt(before, before)

    with pytest.raises(AssertionError):
        _check_opt(before, before,
                   elim_R)


def test_sexp_conversion():
    def f():
        return 10 * (5 + 4)

    sexp = (prim.scalar_mul, 10, (prim.scalar_add, 5, Constant(4)))

    g = sexp_to_graph(sexp)

    assert isomorphic(g, parse(f))


def test_elim():
    def before(x):
        return R(x)

    def after(x):
        return x

    _check_opt(before, after,
               elim_R)


def _idempotent_after(x):
    return P(x)


def test_idempotent():
    def before(x):
        return P(P(x))

    _check_opt(before, _idempotent_after,
               idempotent_P)


def test_idempotent_multi():
    def before(x):
        return P(P(P(P(x))))

    _check_opt(before, _idempotent_after,
               idempotent_P)


def test_idempotent_and_elim():
    def before(x):
        return P(R(P(R(R(P(x))))))

    _check_opt(before, _idempotent_after,
               idempotent_P, elim_R)


def test_multiply_zero():
    def before(x):
        return x * 0

    def after(x):
        return 0

    _check_opt(before, after,
               multiply_by_zero_l, multiply_by_zero_r)


def test_multiply_add_elim_zero():
    def before(x, y):
        return x + y * R(0)

    def after(x, y):
        return x

    _check_opt(before, after,
               elim_R, multiply_by_zero_r, add_zero_r)


def test_replace_twice():
    def before(x):
        return Q(0)

    def after(x):
        return 0

    _check_opt(before, after,
               Q0_to_R, elim_R)


def test_revisit():
    def before(x):
        return Q(P(x))

    def after(x):
        return Q(x)

    _check_opt(before, after,
               QP_to_QR, elim_R)


def test_multi_function():
    def before_helper(x, y):
        return R(x) * R(y)

    def before(x):
        return before_helper(R(x), 3)

    def after_helper(x, y):
        return x * y

    def after(x):
        return after_helper(x, 3)

    _check_opt(before, after,
               elim_R)


def test_closure():
    def before(x):
        y = P(x)

        def sub():
            return Q(y)
        return sub()

    def after(x):

        def sub():
            return Q(x)
        return sub()

    _check_opt(before, after,
               QP_to_QR, elim_R)


def test_closure_2():
    def before(x):
        # Note that y is only accessible through the closure
        y = R(x)

        def sub():
            return y
        return sub()

    def after(x):
        def sub():
            return x
        return sub()

    _check_opt(before, after,
               elim_R)


def test_fn_replacement():
    def before(x):
        return Q(P(P(P(P(x)))))

    def after(x):
        return x

    @pattern_replacer(Q, X)
    def elim_QPs(optimizer, node, equiv):
        # Q(P(...P(x))) => x
        arg = equiv[X]
        while arg.inputs and arg.inputs[0].value == P:
            arg = arg.inputs[1]
        return arg

    _check_opt(before, after,
               elim_QPs)


def test_constant_variable():
    def before(x):
        return Q(15) + Q(x)

    def after(x):
        return P(15) + Q(x)

    Qct_to_P = psub(
        (Q, V),
        (P, V),
        name='Qct_to_P'
    )

    _check_opt(before, after,
               Qct_to_P)
