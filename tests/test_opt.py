
from myia.api import parse
from myia.anf_ir import Graph, Constant
from myia.anf_ir_utils import is_constant, is_apply, is_parameter
from myia.opt import \
    sexp_to_graph, \
    PatternSubstitutionOptimization as psub, \
    PatternOptimizerSinglePass, \
    PatternOptimizerEquilibrium
from myia.unify import Var
from myia.prim import ops as prim, Primitive
from myia.cconv import NestingAnalyzer


X = Var('X')


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
    (prim.mul, 0, X),
    0,
    name='multiply_by_zero_l'
)

multiply_by_zero_r = psub(
    (prim.mul, X, 0),
    0,
    name='multiply_by_zero_r'
)

add_zero_l = psub(
    (prim.add, 0, X),
    X,
    name='add_zero_l'
)

add_zero_r = psub(
    (prim.add, X, 0),
    X,
    name='add_zero_r'
)


def _same_graphs(g1, g2, _equiv=None):
    equiv = dict(zip(g1.parameters, g2.parameters))
    if _equiv:
        equiv.update(_equiv)

    def same(n1, n2):
        if n1 in equiv:
            return equiv[n1] is n2
        if type(n1) is not type(n2):
            return False
        if is_constant(n1):
            return same(n1.value, n2.value)
            # return n1.value == n2.value
        elif is_parameter(n1):
            return False
        elif is_apply(n1):
            success = all(same(i1, i2) for i1, i2 in zip(n1.inputs, n2.inputs))
            if success:
                equiv[n1] = n2
            return success
        elif isinstance(n1, Graph):
            return _same_graphs(n1, n2, equiv)
        else:
            return n1 == n2

    return same(g1.return_, g2.return_)


def _check_opt(before, after):
    gbefore = parse(before)
    gafter = parse(after)

    pass_ = PatternOptimizerSinglePass([
        idempotent_P,
        elim_R,
        Q0_to_R,
        QP_to_QR,
        multiply_by_zero_l,
        multiply_by_zero_r,
        add_zero_l,
        add_zero_r,
    ])

    eq = PatternOptimizerEquilibrium(pass_)

    for g in NestingAnalyzer(gbefore).coverage():
        eq(g)

    assert _same_graphs(gbefore, gafter)


def test_sexp_conversion():
    def f():
        return 10 * (5 + 4)

    sexp = (prim.mul, 10, (prim.add, 5, Constant(4)))

    g = sexp_to_graph(sexp)

    assert _same_graphs(g, parse(f))


def test_elim():
    def before(x):
        return R(x)

    def after(x):
        return x

    _check_opt(before, after)


def _idempotent_after(x):
    return P(x)


def test_idempotent():
    def before(x):
        return P(P(x))

    _check_opt(before, _idempotent_after)


def test_idempotent_multi():
    def before(x):
        return P(P(P(P(x))))

    _check_opt(before, _idempotent_after)


def test_idempotent_and_elim():
    def before(x):
        return P(R(P(R(R(P(x))))))

    _check_opt(before, _idempotent_after)


def test_multiply_zero():
    def before(x):
        return x * 0

    def after(x):
        return 0

    _check_opt(before, after)


def test_multiply_add_elim_zero():
    def before(x, y):
        return x + y * R(0)

    def after(x):
        return x

    _check_opt(before, after)


def test_replace_twice():
    def before(x):
        return Q(0)

    def after(x):
        return 0

    _check_opt(before, after)


def test_revisit():
    def before(x):
        return Q(P(x))

    def after(x):
        return Q(x)

    _check_opt(before, after)


def test_multi_function():
    def before_helper(x, y):
        return R(x) * R(y)

    def before(x):
        return before_helper(R(x), 3)

    def after_helper(x, y):
        return x * y

    def after(x):
        return after_helper(x, 3)

    _check_opt(before, after)


def test_closure():
    def before(x):
        Q  # See issue #47
        y = P(x)

        def sub():
            return Q(y)
        return sub()

    def after(x):
        Q  # See issue #47

        def sub():
            return Q(x)
        return sub()

    _check_opt(before, after)
