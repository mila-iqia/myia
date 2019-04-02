from dataclasses import dataclass
from myia.dtype import Bool, Int, Float
from myia.abstract import ANYTHING as ANY, AbstractScalar, AbstractList, \
    TYPE, VALUE, from_value

B = AbstractScalar({VALUE: ANY, TYPE: Bool})

i16 = AbstractScalar({VALUE: ANY, TYPE: Int[16]})
i32 = AbstractScalar({VALUE: ANY, TYPE: Int[32]})
i64 = AbstractScalar({VALUE: ANY, TYPE: Int[64]})

f16 = AbstractScalar({VALUE: ANY, TYPE: Float[16]})
f32 = AbstractScalar({VALUE: ANY, TYPE: Float[32]})
f64 = AbstractScalar({VALUE: ANY, TYPE: Float[64]})

li16 = AbstractList(i16)
li32 = AbstractList(i32)
li64 = AbstractList(i64)

lf16 = AbstractList(f16)
lf32 = AbstractList(f32)
lf64 = AbstractList(f64)


@dataclass(frozen=True)
class Point:
    x: i64
    y: i64

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __add__(self, other):
        return Point(self.x * other.x, self.y * other.y)


pt = from_value(Point(1, 2), broaden=True)
lpt = AbstractList(pt)
