import pytest

from myia.utils.unify import FilterVar, RestrictedVar, Seq, SVar, \
    Unification, UnificationError, UnionVar, Var, VisitError, \
    expandlist, noseq, svar, uvar, var, PredicateSet


class L(list):
    def __hash__(self):
        return hash(tuple(self))

    def __visit__(self, fn):
        return L(noseq(fn, e) for e in self)


TU = Unification()


@TU.register_visitor(tuple)
def visit_tuple(value, fn):
    return tuple(expandlist(fn(e) for e in value))


def test_Var():
    v1 = var()
    assert isinstance(v1, Var)
    v2 = var()
    assert v1 is not v2
    assert v1 != v2
    assert v1.matches(v2)
    assert v1.matches(object())
    assert str(v1) == v1.tag
    v3 = Var('name')
    assert repr(v3) == 'Var(name)'
    repr(v2)
    assert v1.tag != v2.tag


def test_RestrictedVar():
    v1 = var((2, 3))
    assert isinstance(v1, Var)
    assert isinstance(v1, RestrictedVar)
    v2 = var((2, 3))
    assert v1.matches(v2)
    assert v2.matches(v1)
    v3 = var((1, 2, 3))
    assert not v1.matches(1)
    assert v3.matches(2)
    assert not v1.matches(v3)
    assert v3.matches(v1)
    v4 = var((3, 4))
    v1_4 = v1.intersection(v4)
    assert not v1_4.matches(2)
    assert v1_4.matches(3)
    assert not v1_4.matches(4)
    assert str(v1) == v1.tag
    v5 = var((1, 2))
    assert v1.intersection(v2) is v1
    assert v3.intersection(v2) is v2
    assert v4.intersection(v5) is False
    assert v5.intersection(var()) is NotImplemented
    assert repr(v1) == f'RestrictedVar({v1.tag}, (2, 3))'


def test_FilterVar():

    def floats(v):
        return isinstance(v, float)

    def neg(v):
        return v < 0

    def large(v):
        return abs(v) > 1000

    v1 = var(floats)
    assert isinstance(v1, Var)
    assert isinstance(v1, FilterVar)
    v2 = var(floats)
    assert v1.matches(v2)
    assert v2.matches(v1)
    r1 = var((2.0, 3.0))
    assert v1.matches(r1)
    assert v1.matches(1.0)
    assert not v1.matches(2)
    v3 = var(lambda v: v is None)
    assert not v3.matches(v1)
    assert not v1.matches(v3)
    assert not v3.matches(r1)
    assert v3.matches(None)
    assert not v3.matches(3.0)
    v4 = var(neg)
    v1_4 = v4.intersection(v1)
    assert v4.matches(-1)
    assert not v1_4.matches(-1)
    assert v1_4.matches(-1.0)
    assert not v1_4.matches(1.0)
    v5 = var(large)
    v1_4_5 = v5.intersection(v4).intersection(v1)
    assert v1_4_5.matches(-1111.1)
    assert not v1_4_5.matches(-1111)
    assert not v1_4_5.matches(1.0)
    assert v4.intersection(v4) is v4
    assert v4.intersection(var()) is NotImplemented
    assert str(v1) == v1.tag
    assert repr(v1) == f'FilterVar({v1.tag}, {floats.__name__})'


def test_Seq():
    s = Seq((1, 2, 3))
    assert repr(s) == 'Seq(1, 2, 3)'


def test_SVar():
    sv = SVar()
    assert not sv.matches(1)
    assert not sv.matches((1, 2))
    assert sv.matches(Seq((1,)))
    ssv = str(sv)
    assert ssv == f'*{sv.tag}'
    assert repr(sv) == f'SVar({sv.tag})'
    sv2 = SVar(var(filter=(True, False, 0, 1)))
    assert sv.matches(sv2)
    assert sv2.matches(Seq((True, False, 1)))
    assert not sv2.matches(Seq((1, 2)))


def test_UnionVar():
    uv = UnionVar([(1,), L([1])])

    with pytest.raises(UnificationError):
        uv.matches(None)

    ruv = repr(uv)
    assert ruv == f'UnionVar({uv.tag}, {{[1], (1,)}})'


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
    v = var()
    assert type(v) is Var
    v = var(filter=(None, 0))
    assert v.matches(None)
    assert not v.matches(1)
    v = var(filter=lambda v: isinstance(v, int))
    assert v.matches(2)
    assert not v.matches(2.0)


def test_svar():
    sv = svar()
    assert type(sv) is SVar


def test_uvar():
    uv = uvar((1, 2))
    assert type(uv) is UnionVar
    assert len(uv.values) == 2
    assert 1 in uv.values
    assert 2 in uv.values

    uv2 = uvar([(1, 2), L([1, 2])])
    assert len(uv2.values) == 2

    uv3 = uvar([1, 2, 1, 2])
    assert len(uv3.values) == 2


def test_visit():
    U = Unification()

    def f(x):
        return f

    with pytest.raises(VisitError):
        U.visit(f, None)


def test_clone():
    fv1 = var(filter=lambda v: isinstance(v, int))
    fv2 = TU.clone(fv1)
    assert fv1 is not fv2
    assert fv1.filter is fv2.filter

    rv1 = var(filter=[None, 0])
    rv2 = TU.clone(rv1)
    assert rv1 is not rv2
    assert rv1.legal_values == rv2.legal_values

    sv1 = svar()
    sv2 = TU.clone(sv1)
    assert sv1 is not sv2
    assert isinstance(sv2, SVar)

    v1 = var()
    v2 = var()

    uv1 = uvar([(v1, v2), L([v2])])
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
    v1 = var()
    v2 = var()

    sv1 = svar()

    uv1 = uvar((1, 2))

    dd = {}
    d = TU.unify_union(uv1, 2, dd)
    assert len(d) == 0
    assert d is dd

    with pytest.raises(UnificationError):
        TU.unify_union(uv1, 3, {})

    dd = {}
    uv2 = uvar([(v1,), L([v1])])
    d = TU.unify_union(uv2, v2, dd)
    assert len(d) == 1
    assert d is dd
    assert d[v2].values == uv2.values

    dd = {}
    uv2 = uvar([(v1,), L([v1])])
    d = TU.unify_union(uv2, (v2,), dd)
    assert len(d) == 1
    assert d is dd
    assert d[v1] == v2

    dd = {}
    uv2 = uvar([(v1, v2, sv1), L([v2])])
    d = TU.unify_union(uv2, (1, 2), dd)
    assert len(d) == 3
    assert d is dd
    assert d[v2] == 2

    uv3 = uvar([(v1,), (v1, sv1)])
    with pytest.raises(UnificationError):
        TU.unify_union(uv3, (v2,), {})


def test_unify_raw():
    v1 = var()
    v2 = var()
    v3 = var()

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

    sv1 = svar()
    sv2 = svar()

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

    uv = uvar([(v1,), L([v1])])

    d = TU.unify_raw(uv, (v2,), {})
    assert len(d) == 1
    assert d[v1] is v2

    d = TU.unify_raw((v2,), uv, {})
    assert len(d) == 1
    assert d[v1] is v2


def test_unify_svar():
    v1 = var()
    v2 = var()
    v3 = var()

    sv1 = svar()
    sv2 = svar()

    d = TU.unify_raw(L([v1, (sv1,), (v1, sv1)]), L([v2, v3, (v2, sv2)]),
                     {v3: (1, 2, 3)})
    assert d[sv2] == (1, 2, 3)


# This used to loop infinitely
def test_unify2():
    v1 = var()
    v2 = var()
    assert TU.unify((v1, v1), (v2, v2)) is not None


def test_unify():
    v1 = var()
    v2 = var()
    v3 = var()

    d = TU.unify((v1, v1, v3), (v2, v3, None))
    assert len(d) == 3
    assert d[v1] is None
    assert d[v2] is None
    assert d[v3] is None

    assert TU.unify(None, 0) is None


def test_unify_restrictedvars():
    v1 = RestrictedVar((1, 2))
    v2 = RestrictedVar((2, 3))
    v3 = RestrictedVar((3, 4))

    d = TU.unify((v1, v2), (v2, v1))
    assert d

    vx = d[v1]
    assert vx is not v1
    assert vx is not v2
    assert vx is d[v2]
    assert vx.legal_values == (2,)

    assert not TU.unify(v1, v3)


def test_unify_filtervars():

    def floats(v):
        return isinstance(v, float)

    def neg(v):
        return v < 0

    vf = var(filter=floats)
    vn = var(filter=neg)

    d = TU.unify((vf, vn), (vn, vn))
    assert d

    vfn = d[vf]
    assert vfn is not vf
    assert vfn is not vn
    assert vfn is d[vn]
    assert isinstance(vfn, FilterVar)
    assert vfn.filter == PredicateSet(floats, neg)

    assert TU.unify((vf, vn), (vn, -1.0))
    assert not TU.unify((vf, vn), (vn, -1))
    assert not TU.unify((vf, vn), (vn, 1))


def test_reify():
    v1 = var()
    sv = svar()

    d = {v1: 3.0}
    t = TU.reify(L([v1]), d)
    assert t == L([3.0])

    d = {sv: Seq((3, 4))}
    t = TU.reify((1, 2, sv), d)
    assert t == (1, 2, 3, 4)
