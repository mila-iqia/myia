
import pytest
from myia.utils import Override, Partial, Reset, merge


def test_Partial():

    def f(x, y):
        return x + y

    p1 = Partial(f, x=10)
    p2 = Partial(f, y=20)

    assert p1(y=3) == 13
    assert p2(x=3) == 23
    assert merge(p1, p2)() == 30
    assert merge(p1, {'y': 20})() == 30

    with pytest.raises(TypeError):
        Partial(f, z=10)

    p3 = Reset(Partial(f, y=20))
    with pytest.raises(TypeError):
        merge(p1, p3)()

    p4 = Partial(f, x=[1, 2])
    p5 = Partial(f, x=[3, 4])

    assert merge(p4, p5)(y=[10]) == [1, 2, 3, 4, 10]
    assert merge(p5, p4)(y=[10]) == [3, 4, 1, 2, 10]

    p6 = Override(Partial(f, x=[3, 4]))
    assert merge(p4, p6)(y=[10]) == [3, 4, 10]

    def g(x, y):
        return x * y

    p7 = Partial(g, y=20)
    with pytest.raises(ValueError):
        merge(p1, p7)

    p8 = Override(Partial(g, y=20))
    assert merge(p1, p8)() == 200

    def h(**kw):
        return kw

    p9 = Partial(h, x=10)
    assert p9(y=20, z=30) == dict(x=10, y=20, z=30)

    str(p9), repr(p9)


def test_Partial_class():

    class C:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self):
            return self.x + self.y

    p1 = Partial(C, x=10)
    p2 = Partial(C, y=20)

    assert p1(y=3)() == 13
    assert p2(x=3)() == 23
    assert merge(p1, p2)()() == 30

    with pytest.raises(TypeError):
        Partial(C, z=10)
