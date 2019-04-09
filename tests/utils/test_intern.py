
from dataclasses import dataclass
from myia.utils import Interned, Elements


@dataclass
class Point(metaclass=Interned):
    x: object
    y: object

    def __eqkey__(self):
        return Elements(self, (self.x, self.y))


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
