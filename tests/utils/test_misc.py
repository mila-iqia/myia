import numpy as np
import pytest

from myia import operations
from myia.classes import Cons, Empty
from myia.utils import (
    NS,
    Event,
    Events,
    HasDefaults,
    Named,
    Registry,
    SymbolicKeyInstance,
    newenv,
    smap,
    tags,
)


def test_named():
    named = Named("foo")
    assert repr(named) == "foo"


def test_tags():
    assert tags.Update is tags.Update
    assert repr(tags.Update) == "Update"


@smap.variant
def _sum(self, arg: object, *args):
    return arg + sum(args)


def test_smap():
    assert _sum(10, 20) == 30
    assert _sum(10, 20, 30, 40) == 100
    assert _sum((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert _sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert _sum([(1, [2]), 3], [(4, [5]), 6]) == [(5, [7]), 9]
    assert (_sum(np.ones((2, 2)), np.ones((2, 2))) == np.ones((2, 2)) * 2).all()


def test_smap_failures():
    pairs = [
        ((1, 2), [1, 2]),
        ((1, 2), (1, 2, 3)),
        ((1, 2, 3), (1, 2)),
        (1, [1]),
        ([[1]], [1]),
    ]
    for a, b in pairs:
        with pytest.raises(TypeError):
            _sum(a, b)


def test_event():
    accum = []
    ev = Event("event")

    @ev.register
    def double(ev_, x):
        assert ev_ is ev
        accum.append(x * 2)

    @ev.register
    def triple(ev_, x):
        assert ev_ is ev
        accum.append(x * 3)

    ev(1)
    assert accum == [2, 3]

    ev(100)
    assert accum == [2, 3, 200, 300]

    ev.remove(triple)

    ev(1000)
    assert accum == [2, 3, 200, 300, 2000]


def test_history():
    accum = []

    def history():
        return [1, 2]

    ev = Event("event", history=history)

    @ev.register_with_history
    def push(ev, x):
        accum.append(x)

    assert accum == [1, 2]
    ev(3)
    assert accum == [1, 2, 3]


def test_events():
    accum = []

    def history():
        return [(), ()]

    events = Events(artichoke=None, brownie=history)

    @events.artichoke.register_with_history
    def one(ev):
        assert ev is events.artichoke
        accum.append(1)

    events.artichoke()
    assert accum == [1]

    @events.brownie.register_with_history
    def two(ev):
        assert ev is events.brownie
        accum.append(2)

    assert accum == [1, 2, 2]

    events.brownie()
    assert accum == [1, 2, 2, 2]


def test_events_str_repr():
    ev = Event("event")
    str(ev)
    repr(ev)


def test_NS():
    ns = NS(x=1, y=2)

    assert ns.x == 1
    assert ns.y == 2

    ns.a = 3
    assert ns.a == 3
    assert ns["a"] == 3

    ns["b"] = 4
    assert ns["b"] == 4
    assert ns.b == 4

    assert repr(ns) == "NS(x=1, y=2, a=3, b=4)"


def test_env():
    sk1 = SymbolicKeyInstance("x", 1234)
    sk1b = SymbolicKeyInstance("x", 1234)
    assert sk1 == sk1b

    sk2 = SymbolicKeyInstance("y", 1234)

    e = newenv.set(sk1, 100)
    assert e is not newenv

    assert len(newenv) == 0
    assert len(e) == 1
    assert e.get(sk1, 0) == 100
    assert e.get(sk2, 0) == 0

    e = e.set(sk1b, 200)
    assert len(e) == 1
    assert e.get(sk1, 0) == 200
    assert e.get(sk2, 0) == 0

    e = e.set(sk2, 300)
    assert len(e) == 2
    assert e.get(sk1, 0) == 200
    assert e.get(sk2, 0) == 300


def test_env_add():
    skx = SymbolicKeyInstance("x", 1)
    sky = SymbolicKeyInstance("y", 2)

    ex = newenv.set(skx, 10)
    ey = newenv.set(sky, 20)
    exy = newenv.set(skx, 7).set(sky, 8)

    new = ex.add(ey)
    assert new.get(skx, 0) == 10
    assert new.get(sky, 0) == 20

    new = ey.add(ex)
    assert new.get(skx, 0) == 10
    assert new.get(sky, 0) == 20

    new = exy.add(ex)
    assert new.get(skx, 0) == 17
    assert new.get(sky, 0) == 8


def test_operation_str():
    assert str(operations.user_switch) == repr(operations.user_switch)
    assert str(operations.user_switch) == "myia.operations.user_switch"


def test_list_to_cons():
    li = [1, 2, 3]
    li_c = Cons(1, Cons(2, Cons(3, Empty())))
    assert Cons.from_list(li) == li_c


cantaloup = NS(apple={"banana": 456})


def test_registry():
    r = Registry(default_field="banana")
    a = HasDefaults("a", {"banana": 123}, defaults_field=None)
    b = HasDefaults("b", {"banana": 123}, defaults_field=None)
    c = HasDefaults(
        "c", "tests.utils.test_misc.cantaloup", defaults_field="apple"
    )
    r.register(a)(2)
    assert r[a] == 2
    assert r[b] == 123
    assert r[c] == 456
    with pytest.raises(TypeError):
        print(HasDefaults("d", 123, defaults_field="apple"))
    with pytest.raises(KeyError):
        print(r["xyz"])
