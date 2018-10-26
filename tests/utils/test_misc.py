import pytest
import numpy as np

from myia.utils import Named, TypeMap, smap, Event, Events, NS, Overload, \
    SymbolicKeyInstance, newenv


def test_named():
    named = Named('foo')
    assert repr(named) == 'foo'


@smap.variant
def _sum(self, arg: object, *args):
    return arg + sum(args)


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


def test_Overload_mixins():

    f = Overload()

    @f.register
    def f(t: int):
        return t + 1

    g = Overload()

    @g.register
    def g(t: str):
        return t.upper()

    h = Overload(mixins=[f, g])

    assert f(15) == 16
    with pytest.raises(KeyError):
        f("hello")

    assert g("hello") == "HELLO"
    with pytest.raises(KeyError):
        g(15)

    assert h(15) == 16
    assert h("hello") == "HELLO"


def test_Overload_bootstrap():

    f = Overload().bootstrap()

    assert f.bootstrap() is f

    @f.register
    def f(self, xs: list):
        return [self(x) for x in xs]

    @f.register
    def f(self, x: int):
        return x + 1

    @f.register
    def f(self, x: object):
        return "A"

    assert f([1, 2, "xxx", [3, 4]]) == [2, 3, "A", [4, 5]]

    @f.variant
    def g(self, x: object):
        return "B"

    # This does not interfere with f
    assert f([1, 2, "xxx", [3, 4]]) == [2, 3, "A", [4, 5]]

    # The new method in g is used
    assert g([1, 2, "xxx", [3, 4]]) == [2, 3, "B", [4, 5]]


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
    sk1 = SymbolicKeyInstance('x', {})
    sk1b = SymbolicKeyInstance('x', {})
    assert sk1 == sk1b

    sk2 = SymbolicKeyInstance('y', {})

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
