import gc
import operator
from dataclasses import dataclass

import pytest

from myia.utils.intern import (
    AttrEK,
    CanonStore,
    IncompleteException,
    Interned,
    PossiblyRecursive,
    eq,
    eqrec,
    hash as hsh,
    hashrec,
)


@dataclass(eq=False)
class Point(Interned):
    x: object
    y: object

    def __eqkey__(self):
        return AttrEK(self, ("x", "y"))


class Thingy(Interned, PossiblyRecursive):
    def __init__(self, value):
        self.value = value

    def __eqkey__(self):
        return AttrEK(self, ("value",))


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
    hq = hashrec(a) == hashrec(b)
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

    d = [2]
    d.append(d)

    z1 = []
    z1.append(z1)
    z1.append(z1)

    z2 = []
    z2.append(z2)
    z2.append(z1)

    assert _test([a, b], [b, b])
    assert _test(a, b)
    assert not _test(a, d)
    assert _test(a, [1, a])
    assert _test(a, [1, [1, a]])
    assert _test(b, [1, a])
    assert _test([1, b], [1, a])
    assert _test(z1, z2)
    assert _test(c, [1, c])

    assert _test(Point(1, 2), Point(1, 2))
    assert _test(2 + 9j, 2 + 9j)
    assert _test((1, 2), (1, 2))
    assert not _test((1, 2), (1, 2, 3))
    assert not _test((1, 2), [1, 2])

    assert eq([a, b], [b, b])
    assert hsh([a, b]) == hsh([b, b])


def test_eqrec_incomplete():
    rec1 = Thingy.empty()
    rec2 = Thingy.empty()

    with pytest.raises(IncompleteException):
        _test(rec1, rec2)

    rec1.commit(rec1)
    rec2.commit(rec2)

    rec1 = rec1.intern()
    rec2 = rec2.intern()

    assert _test(rec1, rec2)


def test_canonical():
    t1 = Thingy.new(1)
    t2 = Thingy.new(1)
    assert t1 is not t2
    p1 = Point(t1, 1)
    p2 = Point(t2, 1)
    p3 = Point(t2, 2)
    assert p1 is p2
    assert p1.x is p3.x
    assert p2.x is p3.x


class C:
    pass


def test_weakrefs():
    c = C()
    store = CanonStore(hashfn=hash, eqfn=operator.eq)
    store.set_canonical(c)
    store.gc()
    assert len(store.hashes) == 1
    del c
    gc.collect()
    store.gc()
    assert len(store.hashes) == 0
