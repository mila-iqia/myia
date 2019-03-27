import pytest
import numpy as np

from myia.utils import Named, smap, Event, Events, NS, SymbolicKeyInstance, \
    newenv


def test_named():
    named = Named('foo')
    assert repr(named) == 'foo'


@smap.variant
def _sum(self, arg: object, *args):
    return arg + sum(args)


def test_smap():
    assert _sum(10, 20) == 30
    assert _sum(10, 20, 30, 40) == 100
    assert _sum((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert _sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert _sum([(1, [2]), 3], [(4, [5]), 6]) == [(5, [7]), 9]
    assert (_sum(np.ones((2, 2)), np.ones((2, 2)))
            == np.ones((2, 2))*2).all()


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
    ev = Event('event')

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

    ev = Event('event', history=history)

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
    ev = Event('event')
    str(ev)
    repr(ev)


def test_NS():
    ns = NS(x=1, y=2)

    assert ns.x == 1
    assert ns.y == 2

    ns.a = 3
    assert ns.a == 3
    assert ns['a'] == 3

    ns['b'] = 4
    assert ns['b'] == 4
    assert ns.b == 4

    assert repr(ns) == 'NS(x=1, y=2, a=3, b=4)'


def test_env():
    sk1 = SymbolicKeyInstance('x', 1234)
    sk1b = SymbolicKeyInstance('x', 1234)
    assert sk1 == sk1b

    sk2 = SymbolicKeyInstance('y', 1234)

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
