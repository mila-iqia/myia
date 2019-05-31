
import pytest
import asyncio
import typing

from myia import dtype as ty
from myia.prim import ops as P
from myia.abstract import (
    ANYTHING, MyiaTypeError,
    AbstractScalar, AbstractTuple as T, AbstractList as L,
    AbstractJTagged, AbstractError, AbstractFunction, AbstractUnion,
    InferenceLoop, to_abstract, build_value, amerge,
    Possibilities, PendingFromList,
    VALUE, TYPE, DEAD, find_coherent_result_sync,
    abstract_clone, abstract_clone_async, broaden,
    Pending, concretize_abstract, type_to_abstract,
    InferenceError
)
from myia.utils import SymbolicKeyInstance
from myia.ir import Constant

from .common import Point, to_abstract_test, f32, Ty, af32_of, S, U


def test_to_abstract():
    inst = SymbolicKeyInstance(Constant(123), 456)
    expected = AbstractScalar({VALUE: inst, TYPE: ty.SymbolicKeyType})
    assert to_abstract(inst) == expected


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


def test_amerge():
    a = T([S(1), S(t=ty.Int[64])])
    b = T([S(1), S(t=ty.Int[64])])
    c = T([S(t=ty.Int[64]), S(t=ty.Int[64])])

    assert amerge(a, b, loop=None, forced=False) is a
    assert amerge(a, c, loop=None, forced=False) == c
    assert amerge(c, a, loop=None, forced=False) is c

    with pytest.raises(MyiaTypeError):
        amerge(a, c, loop=None, forced=True)

    assert amerge(1, 2, loop=None, forced=False) is ANYTHING
    with pytest.raises(MyiaTypeError):
        assert amerge(1, 2, loop=None, forced=True)

    with pytest.raises(MyiaTypeError):
        assert amerge("hello", "world", loop=None, forced=False)

    assert amerge(ty.Int, ty.Int[64], loop=None, forced=False) is ty.Int
    assert amerge(ty.Int[64], ty.Int, loop=None, forced=False) is ty.Int
    with pytest.raises(MyiaTypeError):
        amerge(ty.Float, ty.Int, loop=None, forced=False)
    with pytest.raises(MyiaTypeError):
        amerge(ty.Int[64], ty.Int, loop=None, forced=True)

    loop = asyncio.new_event_loop()
    p = PendingFromList([ty.Int[64], ty.Float[64]], None, None, loop=loop)
    assert amerge(ty.Number, p, loop=None, forced=False, bind_pending=False) \
        is ty.Number
    assert amerge(p, ty.Number, loop=None, forced=False, bind_pending=False) \
        is ty.Number
    with pytest.raises(MyiaTypeError):
        print(amerge(p, ty.Number, loop=None, forced=True, bind_pending=False))


def test_merge_possibilities():
    a = Possibilities((1, 2))
    b = Possibilities((2, 3))
    c = Possibilities((2,))
    assert set(amerge(a, b,
                      loop=None,
                      forced=False)) == {1, 2, 3}
    assert amerge(a, c,
                  loop=None,
                  forced=False) is a

    with pytest.raises(MyiaTypeError):
        amerge(a, b, loop=None, forced=True)

    assert amerge(a, c, loop=None, forced=True) is a


def test_merge_from_types():
    a = T([S(1), S(t=ty.Int[64])])

    t1 = type_to_abstract(typing.Tuple)
    t2 = type_to_abstract(typing.Tuple[ty.Int[64], ty.Int[64]])
    t3 = type_to_abstract(typing.Tuple[ty.Int[64]])
    assert amerge(t1, a, loop=None, forced=True) is t1
    assert amerge(t2, a, loop=None, forced=True) is t2
    with pytest.raises(MyiaTypeError):
        amerge(t3, a, loop=None, forced=True)


def test_union():
    a = U(S(t=ty.Int[64]), S(t=ty.Int[32]), S(t=ty.Int[16]))
    b = U(S(t=ty.Int[64]), U(S(t=ty.Int[32]), S(t=ty.Int[16])))
    assert a == b

    c = S(t=ty.Int[64])
    d = U(S(t=ty.Int[64]))
    assert c == d


def test_repr():

    s1 = to_abstract_test(1)
    assert repr(s1) == 'AbstractScalar(Int[64] = 1)'

    s2 = to_abstract_test(f32)
    assert repr(s2) == 'AbstractScalar(Float[32])'

    t1 = to_abstract_test((1, f32))
    assert repr(t1) == f'AbstractTuple((Int[64] = 1, Float[32]))'

    l1 = to_abstract_test([f32])
    assert repr(l1) == f'AbstractList([Float[32]])'

    a1 = to_abstract_test(af32_of(4, 5))
    assert repr(a1) == f'AbstractArray(Float[32] x 4 x 5)'

    p1 = to_abstract_test(Point(1, f32))
    assert repr(p1) == \
        f'AbstractClass(Point(x :: Int[64] = 1, y :: Float[32]))'

    j1 = AbstractJTagged(to_abstract_test(1))
    assert repr(j1) == f'AbstractJTagged(J(Int[64] = 1))'

    ty1 = Ty(f32)
    assert repr(ty1) == 'AbstractType(Ty(Float[32]))'

    e1 = AbstractError(DEAD)
    assert repr(e1) == 'AbstractError(E(DEAD))'

    f1 = AbstractFunction(P.scalar_mul)
    assert repr(f1) == 'AbstractFunction(scalar_mul)'


def test_repr_recursive():
    sa = S(t=ty.Int[64])
    ta = T.empty()
    la = L.empty()
    la.__init__(ta)
    ta.__init__([sa, la])
    ta = ta.intern()
    repr(ta)


@abstract_clone.variant
def upcast(self, x: AbstractScalar, nbits):
    return AbstractScalar({
        VALUE: x.values[VALUE],
        TYPE: ty.Int[nbits],
    })


def test_abstract_clone():
    s1 = S(t=ty.Int[32])
    s2 = S(t=ty.Int[64])
    assert upcast(s1, 64) is s2

    a1 = T([s1, L(s1)])
    assert upcast(a1, 64) is T([s2, L(s2)])


@abstract_clone_async.variant
async def upcast_async(self, x: AbstractScalar):
    return AbstractScalar({
        VALUE: x.values[VALUE],
        TYPE: ty.Int[64],
    })


def test_abstract_clone_async():
    # Coverage test

    async def coro():
        s1 = S(t=ty.Int[32])
        s2 = S(t=ty.Int[64])
        assert (await upcast_async(s1)) is s2

        a1 = T([s1, L(s1)])
        assert (await upcast_async(a1)) is T([s2, L(s2)])

        f1 = AbstractFunction(P.scalar_add, P.scalar_mul)
        assert (await upcast_async(f1)) is f1

        u1 = AbstractUnion([s1])
        assert (await upcast_async(u1)) is AbstractUnion([s2])

    asyncio.run(coro())


def test_broaden_recursive():
    s1 = S(1)
    t1 = T.empty()
    t1.__init__([s1, t1])
    t1 = t1.intern()

    sa = S(t=ty.Int[64])
    ta = T.empty()
    ta.__init__([sa, ta])
    ta = ta.intern()

    assert broaden(t1, None) is ta
    assert broaden(ta, None) is ta

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

    assert broaden(t2, None) is tb
    assert broaden(tb, None) is tb


def test_concretize_recursive():
    loop = asyncio.new_event_loop()
    s = S(t=ty.Int[64])
    p = Pending(None, None, loop=loop)
    t = T([s, p])
    p.set_result(t)

    sa = S(t=ty.Int[64])
    ta = T.empty()
    ta.__init__([sa, ta])
    ta = ta.intern()

    async def coro():
        assert (await concretize_abstract(t)) is ta

    asyncio.run(coro())


def test_find_coherent_result_sync():
    def fn(x):
        if x == 0:
            raise ValueError('Oh no! Zero!')
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
    assert type_to_abstract(bool) is S(t=ty.Bool)
    assert type_to_abstract(typing.List) is L(ANYTHING)
    assert type_to_abstract(typing.Tuple) is T(ANYTHING)
