import pytest

from myia.utils import Named, TypeMap, smap, Event, Events, NS, Overload


def test_named():
    named = Named('foo')
    assert repr(named) == 'foo'


def _sum(*args):
    return sum(args)


def test_typemap():
    tmap = TypeMap()
    tmap.register(int)('int')
    tmap.register(object)('obj')
    assert tmap[int] == 'int'
    assert tmap[str] == 'obj'


def test_typemap_discover():
    def discover(cls):
        return cls.__name__
    tmap = TypeMap({object: '???'}, discover=discover)
    assert tmap[int] == 'int'
    assert tmap[str] == 'str'
    assert tmap[object] == '???'


def test_Overload():
    o = Overload()

    @o.register
    def f(x, y: int):
        return 'int'

    @o.register  # noqa: F811
    def f(x, y: float):
        return 'float'

    assert f(1, 2) == 'int'
    assert f(1, 2.0) == 'float'

    with pytest.raises(Exception):
        @o.register  # noqa: F811
        def f(x: object, y: object):
            return 'too many annotations'

    with pytest.raises(Exception):
        @o.register  # noqa: F811
        def f(x: object, y):
            return 'wrong arg to annotate'

    @o.register  # noqa: F811
    def f(x, y: 'object'):
        return 'object'

    assert f(1, 2) == 'int'
    assert f(1, 2.0) == 'float'
    assert f(1, 'hello') == 'object'


def test_smap():
    assert smap(_sum, 10, 20) == 30
    assert smap(_sum, 10, 20, 30, 40) == 100
    assert smap(_sum, (1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert smap(_sum, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert smap(_sum, [(1, [2]), 3], [(4, [5]), 6]) == [(5, [7]), 9]


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
            smap(_sum, a, b)


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
