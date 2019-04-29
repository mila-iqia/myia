import pytest

from myia.utils import TypeMap, Overload


def test_typemap():
    tmap = TypeMap()
    tmap.register(int)('int')
    tmap.register(object)('obj')
    assert tmap[int] == 'int'
    assert tmap[str] == 'obj'


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
    with pytest.raises(TypeError):
        f("hello")

    assert g("hello") == "HELLO"
    with pytest.raises(TypeError):
        g(15)

    assert h(15) == 16
    assert h("hello") == "HELLO"


def test_Overload_bootstrap():

    f = Overload(bind_to=True)

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


def test_Overload_wrapper():

    f = Overload()

    @f.wrapper
    def f(fn, x):
        return [fn(x)]

    with pytest.raises(TypeError):
        @f.wrapper
        def f(fn, x):
            return [fn(x)]

    @f.register
    def f(x: int):
        return x + 1

    @f.register
    def f(xs: tuple):
        return tuple(f(x) for x in xs)

    assert f(1) == [2]
    assert f((1, 2, (3, 4))) == [([2], [3], [([4], [5])])]


def test_Overload_variant_wrapper():

    f = Overload()

    @f.register
    def f(x: int):
        return x + 1

    assert f(1) == 2

    @f.variant_wrapper
    def g(fn, x):
        return {fn(x)}

    assert g(1) == {2}

    @f.variant_wrapper(initial_state=lambda: 10)
    def h(fn, self, x):
        return self.state * x

    assert h(1) == 10


def test_Overload_stateful():

    f = Overload(initial_state=lambda: -1)

    @f.wrapper
    def f(fn, self, x):
        self.state += 1
        return fn(self, x)

    @f.register
    def f(self, x: type(None)):
        return self.state

    @f.register
    def f(self, xs: tuple):
        return tuple(self(x) for x in xs)

    assert f((None, None)) == (1, 2)
    assert f((None, (None, None))) == (1, (3, 4))
    assert f((None, (None, None))) == (1, (3, 4))

    @f.variant
    def g(self, x: type(None)):
        return self.state * 10

    assert g((None, None)) == (10, 20)
    assert g((None, (None, None))) == (10, (30, 40))
    assert g((None, (None, None))) == (10, (30, 40))

    @f.variant(initial_state=lambda: 0)
    def h(self, x: type(None)):
        return self.state * 100

    assert h((None, None)) == (200, 300)
    assert h((None, (None, None))) == (200, (400, 500))
    assert h((None, (None, None))) == (200, (400, 500))


def test_Overload_repr():

    humptydumpty = Overload()

    @humptydumpty.register
    def humptydumpty(x: int):
        return x

    @humptydumpty.register
    def ignore_this_name(x: str):
        return x

    assert humptydumpty.name.endswith('.humptydumpty')
    r = repr(humptydumpty)
    assert r.startswith('<Overload ')
    assert r.endswith('.humptydumpty>')
