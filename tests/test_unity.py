import pytest

from myia.unity import (Unification, UnificationError, Var, FilterVar, Seq,
                        RestrictedVar, SVar, UnionVar, expandlist, noseq)


class L(list):
    def __hash__(self):
        return hash(tuple(self))


class TUnification(Unification):
    def visit(self, fn, v):
        if isinstance(v, tuple):
            return tuple(expandlist(fn(e) for e in v))

        elif isinstance(v, L):
            return L(noseq(fn, e) for e in v)

        else:
            raise self.VisitError


TU = TUnification()


def test_Var():
    v1 = TU.var('1')
    assert isinstance(v1, Var)
    v2 = TU.var('2')
    assert v1 is not v2
    assert v1 != v2
    assert v1.matches(v2)
    assert v1.matches(object())
    v3 = TU.var('1')
    assert v3 is v1
    assert str(v3) == '1'
    assert repr(v3) == 'Var(1)'


def test_RestrictedVar():
    v1 = TU.var('1', (2, 3))
    assert isinstance(v1, Var)
    assert isinstance(v1, RestrictedVar)
    v2 = TU.var('2', (2, 3))
    assert v1.matches(v2)
    assert v2.matches(v1)
    v3 = TU.var('3', (1, 2, 3))
    assert not v1.matches(1)
    assert v3.matches(2)
    assert not v1.matches(v3)
    assert v3.matches(v1)
    v4 = TU.var('1', (3, 2))
    assert v4 is v1
    v5 = TU.var('1', (2, 1))
    assert v5 is not v1
    assert str(v1) == '1'
    assert repr(v1) == 'RestrictedVar(1, (2, 3))'


def test_FilterVar():
    def floats(v):
        return isinstance(v, float)
    v1 = TU.var('f', floats)
    assert isinstance(v1, Var)
    assert isinstance(v1, FilterVar)
    v2 = TU.var('g', floats)
    assert v1.matches(v2)
    assert v2.matches(v1)
    r1 = TU.var('rv', (2.0, 3.0))
    assert v1.matches(r1)
    assert v1.matches(1.0)
    assert not v1.matches(2)
    v3 = TU.var('b', lambda v: v is None)
    assert not v3.matches(v1)
    assert not v1.matches(v3)
    assert not v3.matches(r1)
    assert v3.matches(None)
    assert not v3.matches(3.0)
    assert str(v1) == 'f'
    assert repr(v1) == f'FilterVar(f, {floats!r})'


def test_Seq():
    s = Seq((1, 2, 3))
    assert repr(s) == 'Seq(1, 2, 3)'


def test_SVar():
    sv = SVar('1')
    assert not sv.matches(1)
    assert not sv.matches((1, 2))
    assert sv.matches(Seq((1,)))
    assert str(sv) == '*1'
    assert repr(sv) == 'SVar(1)'


def test_UnionVar():
    uv = UnionVar('1', [(1,), L([1])])

    with pytest.raises(UnificationError):
        uv.matches(None)

    assert repr(uv) == 'UnionVar(1, {[1], (1,)})'


def test_expandlist():
    assert expandlist((1, 2, 3)) == [1, 2, 3]
    assert expandlist((1, Seq((2, 3, 4)), 5)) == [1, 2, 3, 4, 5]


def test_noseq():
    def f(x):
        return x
    assert noseq(f, 1) == 1

    with pytest.raises(TypeError):
        noseq(f, Seq((2, 3, 4)))

    def f2(x):
        return Seq((x,))

    with pytest.raises(TypeError):
        noseq(f2, 2)


def test_var():
    v = TU.var()
    assert type(v) is Var
    v = TU.var(filter=(None, 0))
    assert v.matches(None)
    assert not v.matches(1)
    v = TU.var(filter=lambda v: isinstance(v, int))
    assert v.matches(2)
    assert not v.matches(2.0)
    with pytest.raises(ValueError):
        TU.var('_1')


def test_svar():
    sv = TU.svar()
    assert type(sv) is SVar


def test_uvar():
    uv = TU.uvar((1, 2))
    assert type(uv) is UnionVar
    assert len(uv.values) == 2
    assert 1 in uv.values
    assert 2 in uv.values

    uv2 = TU.uvar([(1, 2), L([1, 2])])
    assert len(uv2.values) == 2

    uv3 = TU.uvar([1, 2, 1, 2])
    assert len(uv3.values) == 2


def test_visit():
    U = Unification()

    def f(x):
        return f

    with pytest.raises(U.VisitError):
        U.visit(f, None)


def test_clone():
    fv1 = TU.var(filter=lambda v: isinstance(v, int))
    fv2 = TU.clone(fv1)
    assert fv1 is not fv2
    assert fv1.filter is fv2.filter
    assert fv1.tag != fv2.tag

    rv1 = TU.var(filter=[None, 0])
    rv2 = TU.clone(rv1)
    assert rv1 is not rv2
    assert rv1.legal_values == rv2.legal_values
    assert rv1.tag != rv2.tag

    sv1 = TU.svar()
    sv2 = TU.clone(sv1)
    assert sv1 is not sv2
    assert isinstance(sv2, SVar)

    v1 = TU.var()
    v2 = TU.var()

    uv1 = TU.uvar([(v1, v2), L([v2])])
    uv2 = TU.clone(uv1)
    assert uv1 is not uv2
    assert isinstance(uv2, UnionVar)
    assert len(uv2.values) == 2
    vv = None
    for it in uv2.values:
        if isinstance(it, tuple):
            if vv is None:
                vv = it[1]
            else:
                assert vv is it[1]
        elif isinstance(it, L):
            if vv is None:
                vv = it[0]
            else:
                assert vv is it[0]
        else:
            raise AssertionError("Bad item in clone")

    l1 = L([v1])
    l2 = TU.clone(l1)
    assert l1 is not l2
    assert TU.unify(l1, l2)[v1] is l2[0]

    t1 = (v1, v1, v2)
    t2 = TU.clone(t1)
    assert t1 is not t2
    assert t2[0] is t2[1]
    assert t2[0] != t2[2]
    assert len(t2) == 3

    b = TU.clone(None)
    assert b is None

    s1 = Seq((v1, 2, 3))
    s2 = TU.clone(s1)
    assert s1[0] is not s2[0]
    assert s2[1:] == Seq((2, 3))


def test_unify_union():
    v1 = TU.var()
    v2 = TU.var()

    sv1 = TU.svar()

    uv1 = TU.uvar((1, 2))

    dd = {}
    d = TU.unify_union(uv1, 2, dd)
    assert len(d) == 0
    assert d is dd

    with pytest.raises(UnificationError):
        TU.unify_union(uv1, 3, {})

    dd = {}
    uv2 = TU.uvar([(v1,), L([v1])])
    d = TU.unify_union(uv2, v2, dd)
    assert len(d) == 1
    assert d is dd
    assert d[v2].values == uv2.values

    uv3 = TU.uvar([(v1,), (v1, sv1)])
    with pytest.raises(UnificationError):
        TU.unify_union(uv3, (v2,), {})


def test_unify_raw():
    v1 = TU.var('v1')
    v2 = TU.var('v2')
    v3 = TU.var('v3')

    d = TU.unify_raw(v1, None, {})
    assert d[v1] is None

    d = TU.unify_raw(None, v1, {})
    assert d[v1] is None

    d = TU.unify_raw(v1, v2, {})
    assert d[v1] is v2

    d = TU.unify_raw(v1, None, {v1: v2})
    assert d[v2] is None

    d = TU.unify_raw(v1, v3, {v1: v2, v3: None})
    assert d[v2] is None

    d = TU.unify_raw(L([v1]), L([None]), {})
    assert len(d) == 1
    assert d[v1] is None

    d = TU.unify_raw((v1, v1, v3), (v2, v3, None), {})
    assert len(d) == 3
    assert d[v1] == v2
    assert d[v2] == v3
    assert d[v3] is None

    with pytest.raises(UnificationError):
        TU.unify_raw(1, 2, {})

    with pytest.raises(UnificationError):
        TU.unify_raw((v1, v1, v3), (v2, v3), {})

    with pytest.raises(UnificationError):
        TU.unify_raw((v1, v1, v3), L([v2, v2, v3]), {})

    sv1 = TU.svar()
    sv2 = TU.svar()

    d = TU.unify_raw((sv1,), (v1, v2), {})
    assert len(d) == 1
    assert d[sv1] == Seq((v1, v2))

    d = TU.unify_raw((v1, sv1), (v1, v2), {})
    assert len(d) == 1
    assert d[sv1] == Seq((v2,))

    with pytest.raises(UnificationError):
        TU.unify_raw((v1, sv1), (sv2, v2), {})

    with pytest.raises(UnificationError):
        TU.unify_raw((sv1, sv2), (v1, v2), {})

    with pytest.raises(UnificationError):
        TU.unify_raw((v1, v2), (sv1, sv2), {})

    d = TU.unify_raw((v1, sv1), (v2, sv2), {})
    assert len(d) == 2
    assert d[sv1] is sv2
    assert d[v1] is v2

    d = TU.unify_raw((v1, sv1), (v2, v2, v3), {sv1: Seq((v1, v1))})
    assert len(d) == 3
    assert d[v1] is v2
    assert d[v2] is v3

    uv = TU.uvar([(v1,), L([v1])])

    d = TU.unify_raw(uv, (v2,), {})
    assert len(d) == 1
    assert d[v1] is v2

    d = TU.unify_raw((v2,), uv, {})
    assert len(d) == 1
    assert d[v1] is v2


def test_unify_svar():
    v1 = TU.var()
    v2 = TU.var()
    v3 = TU.var()

    sv1 = TU.svar()
    sv2 = TU.svar()

    d = TU.unify_raw(L([v1, (sv1,), (v1, sv1)]), L([v2, v3, (v2, sv2)]),
                     {v3: (1, 2, 3)})
    assert d[sv2] == (1, 2, 3)


# This used to loop infinitely
def test_unify2():
    v1 = TU.var()
    v2 = TU.var()
    assert TU.unify((v1, v1), (v2, v2)) is not None


def test_unify():
    v1 = TU.var('v1')
    v2 = TU.var('v2')
    v3 = TU.var('v3')

    d = TU.unify((v1, v1, v3), (v2, v3, None))
    assert len(d) == 3
    assert d[v1] is None
    assert d[v2] is None
    assert d[v3] is None

    assert TU.unify(None, 0) is None


def test_reify():
    v1 = TU.var('v1')
    v2 = TU.var('v2')
    v3 = TU.var('v3')
    v4 = TU.var('v4')

    d = {v1: 2.0, v2: None, v3: 3, v4: (v1, v3)}
    t = TU.reify(v4, d)
    assert t == (2.0, 3)

    d = {v1: 3.0}
    t = TU.reify(L([v1]), d)
    assert t == L([3.0])
