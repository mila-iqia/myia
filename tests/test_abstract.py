import asyncio
import typing

import numpy as np
import pytest

from myia import xtype as ty
from myia.abstract import (
    ALIASID,
    ANYTHING,
    DEAD,
    TYPE,
    VALUE,
    AbstractBottom,
    AbstractClass,
    AbstractError,
    AbstractFunction,
    AbstractFunctionUnique,
    AbstractHandle,
    AbstractJTagged,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple as T,
    AbstractType,
    AbstractUnion,
    Context,
    InferenceLoop,
    Pending,
    PendingFromList,
    Possibilities,
    TaggedPossibilities,
    TrackDict,
    TransformedFunction,
    abstract_check,
    abstract_clone,
    amerge,
    broaden,
    build_value,
    empty,
    find_coherent_result_sync,
    listof,
    to_abstract,
    type_to_abstract,
)
from myia.classes import Cons, Empty
from myia.ir import Constant
from myia.operations import primitives as P
from myia.pipeline import standard_resources
from myia.utils import (
    InferenceError,
    InternalInferenceError,
    MyiaTypeError,
    SymbolicKeyInstance,
)

from .common import Point, S, Ty, U, af32_of, f32, i16, to_abstract_test


def test_to_abstract_skey():
    inst = SymbolicKeyInstance(Constant(123), 456)
    expected = AbstractScalar({VALUE: inst, TYPE: ty.SymbolicKeyType})
    assert to_abstract(inst) == expected


def test_to_abstract_list():
    assert to_abstract([]) is empty
    assert to_abstract([1, 2, 3]) is listof(S(t=ty.Int[64]))


def test_to_abstract_xtype():
    assert to_abstract(ty.Int[64]) is AbstractType(S(t=ty.Int[64]))


def test_numpy_scalar_to_abstract():
    s1 = AbstractScalar({VALUE: 2, TYPE: i16})
    assert to_abstract_test(np.int16(2)) == s1

    s2 = AbstractScalar({VALUE: 1.5, TYPE: f32})
    assert to_abstract_test(np.float32(1.5)) == s2


def test_build_value():
    assert build_value(S(1)) == 1
    with pytest.raises(ValueError):
        build_value(S(t=ty.Int[64]))
    assert build_value(S(t=ty.Int[64]), default=ANYTHING) is ANYTHING

    assert build_value(T([S(1), S(2)])) == (1, 2)

    loop = InferenceLoop(errtype=Exception)
    p = loop.create_pending(resolve=(lambda: None), priority=(lambda: None))
    with pytest.raises(ValueError):
        assert build_value(S(p, t=ty.Int[64])) is p
    assert build_value(S(p, t=ty.Int[64]), default=ANYTHING) is ANYTHING
    p.set_result(1234)
    assert build_value(S(p, t=ty.Int[64])) == 1234

    pt = Point(1, 2)
    assert build_value(to_abstract_test(pt)) == pt


def test_tagged_possibilities():
    abc = TaggedPossibilities([[1, "a"], [2, "b"], [3, "c"]])
    cab = TaggedPossibilities([[3, "c"], [1, "a"], [2, "b"]])
    assert abc == cab
    assert abc.get(1) == "a"
    with pytest.raises(KeyError):
        abc.get(4)


def test_amerge():
    a = T([S(1), S(t=ty.Int[64])])
    b = T([S(1), S(t=ty.Int[64])])
    c = T([S(t=ty.Int[64]), S(t=ty.Int[64])])

    assert amerge(a, b, forced=False) is a
    assert amerge(a, c, forced=False) == c
    assert amerge(c, a, forced=False) is c

    with pytest.raises(MyiaTypeError):
        amerge(a, c, forced=True)

    assert amerge(1, 2, forced=False) is ANYTHING
    with pytest.raises(MyiaTypeError):
        assert amerge(1, 2, forced=True)

    with pytest.raises(MyiaTypeError):
        assert amerge("hello", "world", forced=False)

    assert amerge(ty.Int, ty.Int[64], forced=False) is ty.Int
    assert amerge(ty.Int[64], ty.Int, forced=False) is ty.Int
    with pytest.raises(MyiaTypeError):
        amerge(ty.Float, ty.Int, forced=False)
    with pytest.raises(MyiaTypeError):
        amerge(ty.Int[64], ty.Int, forced=True)

    assert amerge(
        AbstractError(DEAD), AbstractError(ANYTHING), forced=False
    ) is AbstractError(ANYTHING)

    assert amerge(
        AbstractError(ANYTHING), AbstractError(DEAD), forced=True
    ) is AbstractError(ANYTHING)

    d1 = {"x": 1}
    d2 = {"y": 2}
    with pytest.raises(MyiaTypeError):
        print(amerge(d1, d2))

    td1 = TrackDict({ALIASID: 1})
    td2 = TrackDict({})
    with pytest.raises(MyiaTypeError):
        print(amerge(td1, td2, forced=True))


def test_amerge_pending():
    loop = asyncio.new_event_loop()

    p = PendingFromList([ty.Int[64], ty.Float[64]], None, None, loop=loop)
    assert amerge(ty.Number, p, forced=False, bind_pending=False) is ty.Number
    assert amerge(p, ty.Number, forced=False, bind_pending=False) is ty.Number
    with pytest.raises(MyiaTypeError):
        print(amerge(p, ty.Number, forced=True, bind_pending=False))

    s1 = S(t=ty.Int[32])
    p = Pending(loop=loop, resolve=None, priority=None)
    sp = S(t=p)
    assert amerge(sp, s1, forced=True) is sp
    p.set_result(ty.Int[32])
    assert amerge(sp, s1) is s1
    assert amerge(s1, sp) is s1


def test_merge_possibilities():
    a = Possibilities((1, 2))
    b = Possibilities((2, 3))
    c = Possibilities((2,))
    assert set(amerge(a, b, forced=False)) == {1, 2, 3}
    assert amerge(a, c, forced=False) is a

    with pytest.raises(MyiaTypeError):
        amerge(a, b, forced=True)

    assert amerge(a, c, forced=True) is a


def test_merge_tagged_possibilities():
    abc = TaggedPossibilities([[1, "a"], [2, "b"], [3, "c"]])
    ab = TaggedPossibilities([[1, "a"], [2, "b"]])
    bc = TaggedPossibilities([[2, "b"], [3, "c"]])
    b = TaggedPossibilities([[2, "b"]])
    assert amerge(ab, bc, forced=False) == abc
    assert amerge(ab, b, forced=False) is ab
    assert amerge(b, ab, forced=False) is ab

    with pytest.raises(MyiaTypeError):
        amerge(ab, bc, forced=True)


def test_merge_from_types():
    a = T([S(1), S(t=ty.Int[64])])

    t1 = type_to_abstract(typing.Tuple)
    t2 = type_to_abstract(typing.Tuple[ty.Int[64], ty.Int[64]])
    t3 = type_to_abstract(typing.Tuple[ty.Int[64]])
    assert amerge(t1, a, forced=True) is t1
    assert amerge(t2, a, forced=True) is t2
    with pytest.raises(MyiaTypeError):
        amerge(t3, a, forced=True)


def test_merge_edge_cases():
    a = {1, 2, 3}
    b = {1, 2, 3}
    assert amerge(a, b) is a

    a = AbstractJTagged(ANYTHING)
    b = AbstractJTagged(123)
    assert amerge(a, b) is a

    a = AbstractClass(object, {"x": ANYTHING, "y": ANYTHING})
    b = AbstractClass(object, {"x": 123, "y": ANYTHING})
    assert amerge(a, b) is a


def test_repr():

    s1 = to_abstract_test(1)
    assert repr(s1) == "AbstractScalar(Int[64] = 1)"

    s2 = to_abstract_test(f32)
    assert repr(s2) == "AbstractScalar(Float[32])"

    t1 = to_abstract_test((1, f32))
    assert repr(t1) == f"AbstractTuple((Int[64] = 1, Float[32]))"

    a1 = to_abstract_test(af32_of(4, 5))
    assert repr(a1) == f"AbstractArray(Float[32] x 4 x 5)"

    p1 = to_abstract_test(Point(1, f32))
    assert repr(p1) == f"AbstractClass(Point(x :: Int[64] = 1, y :: Float[32]))"

    j1 = AbstractJTagged(to_abstract_test(1))
    assert repr(j1) == f"AbstractJTagged(J(Int[64] = 1))"

    h1 = AbstractHandle(to_abstract_test(1))
    assert repr(h1) == f"AbstractHandle(H(Int[64] = 1))"

    kw1 = AbstractKeywordArgument("bucket", to_abstract_test(1))
    assert repr(kw1) == f"AbstractKeywordArgument(KW(bucket :: Int[64] = 1))"

    ty1 = Ty(f32)
    assert repr(ty1) == "AbstractType(Ty(AbstractScalar(Float[32])))"

    e1 = AbstractError(DEAD)
    assert repr(e1) == "AbstractError(E(DEAD))"

    f1 = AbstractFunction(P.scalar_mul)
    assert repr(f1) == "AbstractFunction(scalar_mul)"

    fa = AbstractFunction(value=ANYTHING)
    assert repr(fa) == "AbstractFunction(ANYTHING)"

    tu1 = AbstractTaggedUnion([[13, s2], [4, to_abstract_test(i16)]])
    assert repr(tu1) == "AbstractTaggedUnion(U(4 :: Int[16], 13 :: Float[32]))"

    bot = AbstractBottom()
    assert repr(bot) == "AbstractBottom(⊥)"


def test_repr_recursive():
    sa = S(t=ty.Int[64])
    ta = T.empty()
    la = T.empty()
    la.__init__([ta])
    ta.__init__([sa, la])
    ta = ta.intern()
    repr(ta)


@abstract_clone.variant
def upcast(self, x: AbstractScalar, *, nbits):
    return AbstractScalar({VALUE: x.xvalue(), TYPE: ty.Int[nbits]})


def test_abstract_clone():
    s1 = S(t=ty.Int[32])
    s2 = S(t=ty.Int[64])
    assert upcast(s1, nbits=64) is s2

    a1 = T([s1, AbstractClass(object, {"field": s1})])
    a2 = T([s2, AbstractClass(object, {"field": s2})])
    assert upcast(a1, nbits=64) is a2

    jt = TransformedFunction(AbstractFunctionUnique((s1,), s1), P.J)
    assert upcast(jt, nbits=64).fn.args == [s2]
    assert upcast(jt, nbits=64).fn.output is s2


def test_abstract_clone_pending():
    loop = asyncio.new_event_loop()
    s1 = S(t=ty.Int[32])
    p = Pending(loop=loop, resolve=None, priority=None)
    sp = S(t=p)
    assert abstract_clone(sp) is sp
    p.set_result(ty.Int[32])
    assert abstract_clone(sp) is s1


def test_abstract_clone_edge_cases():
    tup = T(ANYTHING)
    assert abstract_check(tup)
    tup2 = abstract_clone(tup)
    assert tup is tup2


def test_broaden_recursive():
    s1 = S(1)
    t1 = T.empty()
    t1.__init__([s1, t1])
    t1 = t1.intern()

    sa = S(t=ty.Int[64])
    ta = T.empty()
    ta.__init__([sa, ta])
    ta = ta.intern()

    assert broaden(t1) is ta
    assert broaden(ta) is ta

    t2 = T.empty()
    u2 = AbstractUnion.empty()
    u2.__init__([s1, t2])
    t2.__init__([s1, u2])
    t2 = t2.intern()

    tb = T.empty()
    ub = AbstractUnion.empty()
    ub.__init__([sa, tb])
    tb.__init__([sa, ub])
    tb = tb.intern()

    assert broaden(t2) is tb
    assert broaden(tb) is tb


def test_find_coherent_result_sync():
    def fn(x):
        if x == 0:
            raise ValueError("Oh no! Zero!")
        else:
            return x > 0

    loop = asyncio.new_event_loop()
    p1 = PendingFromList([1, 2, 3], None, None, loop=loop)
    p2 = PendingFromList([-1, -2, -3], None, None, loop=loop)
    p3 = PendingFromList([1, 2, -3], None, None, loop=loop)
    p4 = PendingFromList([0], None, None, loop=loop)
    assert find_coherent_result_sync(p1, fn) is True
    assert find_coherent_result_sync(p2, fn) is False
    with pytest.raises(InferenceError):
        find_coherent_result_sync(p3, fn)
    with pytest.raises(ValueError):
        find_coherent_result_sync(p4, fn)

    p = Pending(None, None, loop=loop)
    with pytest.raises(InferenceError):
        find_coherent_result_sync(p, fn)

    assert find_coherent_result_sync(10, fn) is True
    assert find_coherent_result_sync(-10, fn) is False


def test_type_to_abstract():
    assert type_to_abstract(int) is S(t=ty.Int[64])
    assert type_to_abstract(float) is S(t=ty.Float[64])
    assert type_to_abstract(bool) is S(t=ty.Bool)
    assert type_to_abstract(typing.List) is U(
        type_to_abstract(Empty), type_to_abstract(Cons)
    )
    assert type_to_abstract(typing.Tuple) is T(ANYTHING)


def test_get_resolved():
    # eng = standard_pipeline.make().resources.inferrer.engine
    eng = standard_resources().inferrer.engine
    ref = eng.ref(Constant(123), Context.empty())
    with pytest.raises(InternalInferenceError):
        ref.get_resolved()
