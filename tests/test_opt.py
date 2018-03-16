
from myia.api import parse
from myia.anf_ir import Constant, Apply, Parameter
from myia.opt import *
from myia import primops as prim
from myia.primops import Primitive


# We will optimize patterns of these fake primitives

P = Primitive('P')
Q = Primitive('Q')
R = Primitive('R')


idempotent_P = simple_sub(
    (P, (P, X)),
    (P, X)
)

elim_R = simple_sub(
    (R, X),
    X
)

Q0_to_R = simple_sub(
    (Q, 0),
    (R, 0)
)

QP_to_QR = simple_sub(
    (Q, (P, X)),
    (Q, (R, X))
)

multiply_by_zero_l = simple_sub(
    (prim.mul, 0, X),
    0
)

multiply_by_zero_r = simple_sub(
    (prim.mul, X, 0),
    0
)

add_zero_l = simple_sub(
    (prim.add, 0, X),
    X
)

add_zero_r = simple_sub(
    (prim.add, X, 0),
    X
)


def _check_same(g1, g2):
    equiv = dict(zip(g1.parameters, g2.parameters))

    def same(n1, n2):
        if n1 in equiv:
            return equiv[n1] is n2
        if type(n1) is not type(n2):
            return False
        if isinstance(n1, Constant):
            return n1.value == n2.value
        elif isinstance(n1, Parameter):
            return False
        else:
            success = all(same(i1, i2) for i1, i2 in zip(n1.inputs, n2.inputs))
            if success:
                equiv[n1] = n2
            return success

    return same(g1.return_, g2.return_)


def _check_opt(opt, before, after):
    gbefore = parse(before)
    gafter = parse(after)

    eqtrans = EquilibriumTransformer(
        None,
        {gbefore},
        [
            idempotent_P,
            elim_R,
            Q0_to_R,
            QP_to_QR,
            multiply_by_zero_l,
            multiply_by_zero_r,
            add_zero_l,
            add_zero_r,
        ]
    )

    eqtrans.run()

    assert _check_same(gbefore, gafter)


def test_elim():
    def before(x):
        return R(x)

    def after(x):
        return x

    _check_opt(None, before, after)


def _idempotent_after(x):
    return P(x)


def test_idempotent():
    def before(x):
        return P(P(x))

    _check_opt(None, before, _idempotent_after)


def test_idempotent_multi():
    def before(x):
        return P(P(P(P(x))))

    _check_opt(None, before, _idempotent_after)


def test_idempotent_and_elim():
    def before(x):
        return P(R(P(R(R(P(x))))))

    _check_opt(None, before, _idempotent_after)


def test_multiply_zero():
    def before(x):
        return x * 0

    def after(x):
        return 0

    _check_opt(None, before, after)


def test_multiply_add_elim_zero():
    def before(x, y):
        return x + y * R(0)

    def after(x):
        return x

    _check_opt(None, before, after)


def test_replace_twice():
    def before(x):
        return Q(0)

    def after(x):
        return 0

    _check_opt(None, before, after)


def test_revisit():
    def before(x):
        return Q(P(x))

    def after(x):
        return Q(x)

    _check_opt(None, before, after)
