
from dataclasses import dataclass
from myia.dtype import Bool, Int, UInt, Float, List, Array, Tuple, Function, \
    Object, pytype_to_myiatype
from myia.ir import MultitypeGraph


B = Bool
L = List
A = Array
T = Tuple
F = Function

i16 = Int[16]
i32 = Int[32]
i64 = Int[64]

u64 = UInt[64]

f16 = Float[16]
f32 = Float[32]
f64 = Float[64]

li16 = L[Int[16]]
li32 = L[Int[32]]
li64 = L[Int[64]]

lf16 = L[Float[16]]
lf32 = L[Float[32]]
lf64 = L[Float[64]]

ai16 = A[i16]
ai32 = A[i32]
ai64 = A[i64]

af16 = A[f16]
af32 = A[f32]
af64 = A[f64]

Nil = T[()]


@dataclass(frozen=True)
class Thing:
    contents: Object


@dataclass(frozen=True)
class Point:
    x: i64
    y: i64

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


@dataclass(frozen=True)
class Point3D:
    x: i64
    y: i64
    z: i64

    def abs(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


Point_t = pytype_to_myiatype(Point)
Point3D_t = pytype_to_myiatype(Point3D)
Thing_f = pytype_to_myiatype(Thing, Thing(1.0))
Thing_ftup = pytype_to_myiatype(Thing, Thing((1.0, 2.0)))


mysum = MultitypeGraph('mysum')


@mysum.register(i64)
def _mysum1(x):
    return x


@mysum.register(i64, i64)
def _mysum2(x, y):
    return x + y


@mysum.register(i64, i64, i64)
def _mysum3(x, y, z):
    return x + y + z
