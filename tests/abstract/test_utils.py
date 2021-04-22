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

from ..common import A, Un, one_test_per_assert

o0 = data.Opaque(0)
o1 = data.Opaque(1)
o2 = data.Opaque(2)
o3 = data.Opaque(3)

ph1 = data.Placeholder()
ph2 = data.Placeholder()


def test_ph_different():
    assert ph1 is not ph2


@one_test_per_assert
def test_is_concrete():
    assert is_concrete(A(1))
    assert is_concrete(A(1, 2))
    assert not is_concrete(o1)
    assert not is_concrete(A(1, o1))
    assert not is_concrete(ph1)
    assert not is_concrete(A(1, ph1))


@one_test_per_assert
def test_canonical():
    assert canonical(ph1, mapping={}) == o0
    assert canonical(ph2, mapping={}) == o0

    assert canonical(A(ph1, int), mapping={}) is A(o0, int)
    assert canonical(A(ph2, int), mapping={}) is A(o0, int)
    assert canonical(A(ph1, ph2), mapping={}) is A(o0, o1)
    assert canonical(A(ph2, ph1), mapping={}) is A(o0, o1)
    assert canonical(A(ph1, ph1), mapping={}) is A(o0, o0)

    assert canonical(A((ph2, ph2), (ph1, ph1)), mapping={}) is A(
        (o0, o0), (o1, o1)
    )

    assert canonical(A(o0, o1), mapping={}) is A(o0, o1)
    assert canonical(A(o1, o0), mapping={}) is A(o0, o1)


def test_fresh_generics():
    assert fresh_generics(o0, mapping={}) is not fresh_generics(o0, mapping={})

    tu1 = A(o0, o0)
    tu1_unc = fresh_generics(tu1, mapping={})
    assert tu1_unc.elements[0] is tu1_unc.elements[1]

    tu2 = A(o0, o1)
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
        A(o0, o1),
        A(o1, o0),
    ]:
        mapping = {}
        canon = canonical(x, mapping=mapping)
        assert uncanonical(canon, mapping=mapping) is x

        mapping = {}
        fgn = fresh_generics(x, mapping=mapping)
        assert uncanonical(fgn, mapping=mapping) is x


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
    assert _utest(o0, A(1), A(1), {o0: A(1)})
    assert _utest(A(1), o0, A(1), {o0: A(1)})
    assert _utest(
        A(o0, 1, o1), A(3, o2, 7), A(3, 1, 7), {o0: A(3), o1: A(7), o2: A(1)}
    )
    assert _utest(
        A(o0, o1, o2), A(8, o0, o1), A(8, 8, 8), {o0: A(8), o1: A(8), o2: A(8)}
    )

    assert _utest(A(o0, o0), A(6, o0), A(6, 6))

    # Unions
    assert _utest(Un(o0), Un(6), Un(6))
    assert _utest(Un(o0, o0), Un(6, o0), Un(6, 6))

    # Variables remain
    assert _utest(o0, o1)


@one_test_per_assert
def test_cannot_unify():
    assert not _utest(A(1), A(2))
    assert not _utest(A(o0, 3), A(7, o0))
    assert not _utest(A(o0, o1, o2, 9), A(7, o0, o1, o2))
    assert not _utest(Un(1), Un(2))
    assert not _utest(Un(1), Un(1, 2))


def test_unify_recursive():
    res, U = unify(o0, A(3, o0))
    assert res.elements[0] is A(3)
    assert res.elements[1] is res

    res, U = unify(A(o0, o1), A((7, o1), (8, o0)))
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
    assert _mtest(o0, A(1), A(1), {o0: A(1)})
    assert _mtest(Un(1), Un(2), Un(1, 2))
    assert _mtest(Un(o0), Un(6), Un(o0, 6))
    assert _mtest(Un(1), Un(1, 2), Un(1, 1, 2))


@one_test_per_assert
def test_get_generics():
    assert get_generics(A(o0, 1, o1)) == {o0, o1}
    assert get_generics(A(int, float)) == set()
