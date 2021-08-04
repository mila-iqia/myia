from myia.abstract import data
from myia.abstract.utils import (
    MapError,
    canonical,
    fresh_generics,
    get_generics,
    is_concrete,
    merge,
    uncanonical,
    unify,
)
from myia.testing.common import A, Un

from ..common import one_test_per_assert

cg0 = data.CanonGeneric(0)
cg1 = data.CanonGeneric(1)
cg2 = data.CanonGeneric(2)

ph1 = data.Placeholder()
ph2 = data.Placeholder()


def test_ph_different():
    assert ph1 is not ph2


@one_test_per_assert
def test_is_concrete():
    assert is_concrete(A(1))
    assert is_concrete(A(1, 2))
    assert not is_concrete(cg1)
    assert not is_concrete(A(1, cg1))
    assert not is_concrete(ph1)
    assert not is_concrete(A(1, ph1))


@one_test_per_assert
def test_canonical():
    assert canonical(ph1, mapping={}) == cg0
    assert canonical(ph2, mapping={}) == cg0

    assert canonical(A(ph1, int), mapping={}) is A(cg0, int)
    assert canonical(A(ph2, int), mapping={}) is A(cg0, int)
    assert canonical(A(ph1, ph2), mapping={}) is A(cg0, cg1)
    assert canonical(A(ph2, ph1), mapping={}) is A(cg0, cg1)
    assert canonical(A(ph1, ph1), mapping={}) is A(cg0, cg0)

    assert canonical(A((ph2, ph2), (ph1, ph1)), mapping={}) is A(
        (cg0, cg0), (cg1, cg1)
    )

    assert canonical(A(cg0, cg1), mapping={}) is A(cg0, cg1)
    assert canonical(A(cg1, cg0), mapping={}) is A(cg0, cg1)


def test_fresh_generics():
    assert fresh_generics(cg0, mapping={}) is not fresh_generics(
        cg0, mapping={}
    )

    tu1 = A(cg0, cg0)
    tu1_unc = fresh_generics(tu1, mapping={})
    assert tu1_unc.elements[0] is tu1_unc.elements[1]

    tu2 = A(cg0, cg1)
    tu2_unc = fresh_generics(tu2, mapping={})
    assert tu2_unc.elements[0] is not tu2_unc.elements[1]


def test_uncanonical():
    for x in [
        ph1,
        ph2,
        A(ph1, int),
        A(ph2, int),
        A(ph1, ph2),
        A(ph2, ph1),
        A(ph1, ph1),
        A((ph2, ph2), (ph1, ph1)),
        A(cg0, cg1),
        A(cg1, cg0),
    ]:
        mapping = {}
        canon = canonical(x, mapping=mapping)
        assert uncanonical(canon, mapping=mapping) is x

        mapping = {}
        fgn = fresh_generics(x, mapping=mapping)
        assert uncanonical(fgn, mapping=mapping) is x


def test_uncanonical_untransformed():
    assert uncanonical(ph1, mapping={}) is ph1


def _utest(a, b, expected=True, mappings={}):
    try:
        c, U = unify(a, b)
    except MapError:
        return False
    if expected is not True:
        assert c == expected
    for entry, canon in mappings.items():
        print(entry, canon, U.canon[entry])
        assert U.canon[entry] == canon
    return True


@one_test_per_assert
def test_unify():
    assert _utest(cg0, A(1), A(1), {cg0: A(1)})
    assert _utest(A(1), cg0, A(1), {cg0: A(1)})
    assert _utest(
        A(cg0, 1, cg1),
        A(3, cg2, 7),
        A(3, 1, 7),
        {cg0: A(3), cg1: A(7), cg2: A(1)},
    )
    assert _utest(
        A(cg0, cg1, cg2),
        A(8, cg0, cg1),
        A(8, 8, 8),
        {cg0: A(8), cg1: A(8), cg2: A(8)},
    )

    assert _utest(A(cg0, cg0), A(6, cg0), A(6, 6))

    # Unions
    assert _utest(Un(cg0), Un(6), Un(6))
    assert _utest(Un(cg0, cg0), Un(6, cg0), Un(6, 6))

    # Variables remain
    assert _utest(cg0, cg1)

    # Transferred from old `test_cannot_unify`
    # Value tracks are now unified with ANYTHING if values are different.
    # So, these tests pass now.
    assert _utest(A(1), A(2))
    assert _utest(A(cg0, 3), A(7, cg0))
    assert _utest(A(cg0, cg1, cg2, 9), A(7, cg0, cg1, cg2))
    assert _utest(Un(1), Un(2))


@one_test_per_assert
def test_cannot_unify():
    assert not _utest(Un(1), Un(1, 2))


def test_unify_recursive():
    res, U = unify(cg0, A(3, cg0))
    assert res.elements[0] is A(3)
    assert res.elements[1] is res

    res, U = unify(A(cg0, cg1), A((7, cg1), (8, cg0)))
    a = res.elements[0]
    b = res.elements[1]
    assert a.elements[0] is A(7)
    assert a.elements[1] is b
    assert b.elements[0] is A(8)
    assert b.elements[1] is a


def _mtest(a, b, expected=True, mappings={}):
    try:
        c, U = merge(a, b)
    except MapError:
        return False
    if expected is not True:
        assert c == expected
    for entry, canon in mappings.items():
        print(entry, canon, U.canon[entry])
        assert U.canon[entry] == canon
    return True


@one_test_per_assert
def test_merge():
    assert _mtest(cg0, A(1), A(1), {cg0: A(1)})
    assert _mtest(Un(1), Un(2), Un(1, 2))
    assert _mtest(Un(cg0), Un(6), Un(cg0, 6))
    assert _mtest(Un(1), Un(1, 2), Un(1, 1, 2))


@one_test_per_assert
def test_get_generics():
    assert get_generics(A(cg0, 1, cg1)) == {cg0, cg1}
    assert get_generics(A(int, float)) == set()
