
import pytest
from dataclasses import dataclass
from myia.utils import Interned, Elements, eqrec, hashrec, eq, hash as hsh, \
    IncompleteException


@dataclass(frozen=True)
class Point(Interned):
    x: object
    y: object

    def __eqkey__(self):
        return Elements(self, (self.x, self.y))


class Thingy(Interned):
    def __init__(self, value=None):
        if value is None:
            self._incomplete = True
        else:
            self.set(value)

    def set(self, value):
        self.value = value
        self._incomplete = False

    def __eqkey__(self):
        return Elements(self, (self.value,))


def test_interned():
    p1 = Point(10, 20)
    p2 = Point(10, 20)
    p3 = Point(20, 30)

    assert p1 is p2
    assert p1 is not p3
    assert p1.x == 10
    assert p1.y == 20

    p4 = Point.new(10, 20)

    assert p4 is not p1
    assert p4.intern() is p1

    p5 = Point(p1, p3)
    p6 = Point(p4, p3)

    assert p5 is p6


def _test(a, b):
    eq = eqrec(a, b, cache=set())
    hq = hashrec(a, frozenset(), {}) == hashrec(b, frozenset(), {})
    # This is a sanity check that assumes no collisions
    assert hq == eq
    return eq


def test_eqrec():

    a = [1]
    a.append(a)

    b = [1]
    b.append(b)

    c = [1, [1]]
    c[1].append(c)

    z1 = []
    z1.append(z1)
    z1.append(z1)

    z2 = []
    z2.append(z2)
    z2.append(z1)

    assert _test([a, b], [b, b])
    assert _test(a, b)
    assert not _test(a, [1, a])
    assert not _test(a, [1, [1, a]])
    assert not _test(b, [1, a])
    assert _test([1, b], [1, a])
    assert not _test(z1, z2)
    assert not _test(c, [1, c])

    assert _test(Point(1, 2), Point(1, 2))
    assert _test(2+9j, 2+9j)
    assert _test((1, 2), (1, 2))
    assert not _test((1, 2), (1, 2, 3))
    assert not _test((1, 2), [1, 2])

    assert eq([a, b], [b, b])
    assert hsh([a, b]) == hsh([b, b])


def test_eqrec_incomplete():
    rec1 = Thingy()
    rec2 = Thingy()

    with pytest.raises(IncompleteException):
        _test(rec1, rec2)

    rec1.set(rec1)
    rec2.set(rec2)

    rec1 = rec1.intern()
    rec2 = rec2.intern()

    assert _test(rec1, rec2)
