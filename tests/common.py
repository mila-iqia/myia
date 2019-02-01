
import numpy as np
from dataclasses import dataclass, is_dataclass
from myia import dtype
from myia.abstract.base import VALUE, TYPE, SHAPE, \
    AbstractBase, AbstractScalar, AbstractArray, concretize_abstract, \
    AbstractList, AbstractTuple, AbstractType, AbstractClass, \
    AbstractJTagged
from myia.dtype import Bool, Int, UInt, Float, List, Array, Tuple, Function, \
    Object, pytype_to_myiatype, TypeType
from myia.abstract import ANYTHING
from myia.ir import MultitypeGraph
from myia.utils import overload, EnvInstance


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


###########################
# Abstract value builders #
###########################


def arr_of(t, shp, value):
    return AbstractArray(AbstractScalar({
        VALUE: value,
        TYPE: t,
    }), {SHAPE: shp})


def ai64_of(*shp, value=ANYTHING):
    return arr_of(i64, shp, value)


def ai32_of(*shp, value=ANYTHING):
    return arr_of(i32, shp, value)


def af64_of(*shp, value=ANYTHING):
    return arr_of(f64, shp, value)


def af32_of(*shp, value=ANYTHING):
    return arr_of(f32, shp, value)


def af16_of(*shp, value=ANYTHING):
    return arr_of(f16, shp, value)


def JT(a):
    return AbstractJTagged(to_abstract(a))


def S(x, t=None):
    return AbstractScalar({
        VALUE: x,
        TYPE: t or dtype.pytype_to_myiatype(type(x)),
    })


def Shp(*vals):
    return to_abstract(tuple(S(v, u64) for v in vals))


def Ty(t):
    # TODO: fix interface
    return AbstractType(t)


@overload(bootstrap=True)
def to_abstract(self, x: (bool, int, float, str, EnvInstance)):
    return AbstractScalar({
        VALUE: x,
        TYPE: dtype.pytype_to_myiatype(type(x)),
    })


@overload  # noqa: F811
def to_abstract(self, x: (dtype.Number, dtype.Bool,
                          dtype.External, dtype.EnvType)):
    return AbstractScalar({VALUE: ANYTHING, TYPE: x})


@overload  # noqa: F811
def to_abstract(self, x: np.ndarray):
    return AbstractArray(
        AbstractScalar({
            VALUE: ANYTHING,
            TYPE: dtype.np_dtype_to_type(str(x.dtype)),
        }),
        {SHAPE: x.shape}
    )


@overload  # noqa: F811
def to_abstract(self, x: AbstractBase):
    return x


@overload  # noqa: F811
def to_abstract(self, tup: tuple):
    return AbstractTuple([self(x) for x in tup])


@overload  # noqa: F811
def to_abstract(self, l: list):
    assert len(l) == 1
    return AbstractList(self(l[0]))


@overload  # noqa: F811
def to_abstract(self, x: Exception):
    return x


@overload  # noqa: F811
def to_abstract(self, t: type):
    return self[t](t)


@overload  # noqa: F811
def to_abstract(self, x: object):
    if is_dataclass(x):
        typ = dtype.pytype_to_myiatype(type(x), x)
        new_args = {}
        for name, field in x.__dataclass_fields__.items():
            new_args[name] = self(getattr(x, name))
        return AbstractClass(typ.tag, new_args, typ.methods)
    else:
        raise Exception(f'Cannot convert: {x}')


###############
# Dataclasses #
###############


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


class FixedMatrix:
    """Utility class to create matrices for tests."""

    def __init__(self, array):
        """Initialize a FixedMatrix."""
        self.array = np.array(array)

    def __call__(self, r, c, dtype='float64'):
        """Create a r by c matrix."""
        return self.array[:r, :c].astype(dtype)

    def __mul__(self, factor):
        """Scale the matrix by a factor."""
        return FixedMatrix(self.array * factor)


MA = FixedMatrix([
    [-0.58, -5.05,  2.24,  0.31, -4.81, -3.07, -2.03, -9.75,  0.54,  8.24],
    [-2.58,  3.23, -1.85,  8.19,  2.33, -8.83, -1.36, -4.95,  2.58, -0.01],
    [-7.85,  7.39, -7.44,  1.65,  5.39,  2.08, -9.55,  8.62, -6.10,  2.81],
    [-3.18, -7.83,  0.90, -8.60,  5.77, -5.81, -9.31, -2.42,  0.27, -6.05],
    [-2.34, -1.19,  2.79,  4.20,  6.36,  2.05,  1.86, -2.63,  5.08,  5.08],
    [+7.98,  3.87,  6.91, -0.68,  6.63,  2.70,  4.25, -4.50, -9.43, -6.91],
    [+3.51, -1.29,  3.88, -6.92,  6.01,  1.99,  3.17,  4.11, -2.36, -6.81],
    [-7.84,  8.26,  4.29,  0.86, -6.85,  2.64, -0.09, -8.95,  4.63, -0.30],
    [+3.18,  1.69, -9.50, -2.80, -9.43,  5.68, -7.71,  1.46,  7.33, -8.09],
    [+9.98,  3.43, -7.26,  3.12,  8.42,  2.72,  0.73, -2.31,  5.70, -2.14]
])


MB = FixedMatrix([
    [-5.47, -1.32, -8.49,  5.57,  8.16, -4.40, -3.73,  1.04,  0.05,  2.13],
    [+1.53,  1.09,  9.36,  5.46,  1.12,  2.20, -0.02, -5.71, -2.72, -4.98],
    [-2.85,  9.24,  8.35,  4.44, -4.14, -1.04, -1.31, -2.50, -5.22, -5.52],
    [+8.16, -0.06, -5.78, -3.96, -3.85, -6.94, -8.73,  0.18,  3.05,  9.51],
    [+8.25, -7.42, -6.05,  4.24,  5.84, -0.50, -3.43, -6.15,  3.73, -8.30],
    [+2.90, -8.07,  7.25, -0.95,  1.34,  7.17, -4.91,  5.63,  3.79,  6.85],
    [+0.72, -5.83, -0.67,  1.09,  0.27,  5.61, -2.07, -7.71,  5.75,  1.68],
    [-3.86, -2.87, -9.72, -5.03, -7.14,  4.64, -8.48, -0.45,  9.89,  4.83],
    [-4.51, -4.35, -3.09,  6.20,  5.77, -8.05,  4.36,  3.11,  2.82, -1.36],
    [-9.24, -8.71, -1.48, -7.69, -6.03, -1.66, -4.01,  4.05,  5.47,  5.63]
])


MC = FixedMatrix([
    [+9.74, -7.26,  3.22, -0.87,  0.07,  8.87,  1.92,  4.18, -0.08, -8.94],
    [-3.22,  1.36,  2.01,  6.78,  1.83,  0.48,  5.87, -1.07,  5.28,  4.33],
    [+7.23, -8.37, -9.51, -7.43, -8.51,  7.22,  8.85,  6.20,  2.14, -6.20],
    [-1.00,  8.82,  2.83,  7.55, -7.12,  9.97, -8.97, -2.26, -0.91, -5.74],
    [+4.12, -9.56, -7.77,  5.61,  0.69,  1.31,  1.05,  0.38,  5.15, -8.19],
    [-3.30, -6.66,  0.61, -4.82,  8.17,  8.93, -7.44, -6.85, -6.72, -3.13],
    [-3.12, -2.06, -4.35, -4.20,  7.08,  9.47, -7.89, -9.80,  1.00, -6.11],
    [+2.68,  0.30, -5.70,  0.77,  8.08, -7.33,  7.04, -3.54, -1.62,  4.61],
    [+7.03,  5.90, -2.47,  9.47, -4.04,  3.68, -3.94,  2.16, -5.65, -2.32],
    [-4.62,  9.66,  4.98, -0.60, -0.53,  9.27, -4.52,  9.43,  9.61, -6.20]
])


MD = FixedMatrix([
    [+1.12,  8.41, -0.83, -8.09, -7.82, -5.72,  4.89, -3.93, -4.30,  7.83],
    [+2.00, -8.60,  3.05, -8.27,  7.40,  9.60,  6.49, -1.46, -8.79, -0.09],
    [+4.68,  3.10, -4.34, -4.75,  7.16, -0.48,  0.43,  2.11, -9.19,  4.81],
    [-6.36,  2.01, -1.93, -6.87, -2.75,  9.09, -8.27, -2.62, -7.86,  0.62],
    [+5.86,  1.84,  0.84, -8.05, -7.42,  4.41,  0.00, -0.57,  1.72, -5.59],
    [+5.74, -7.60, -9.21,  1.28, -8.11,  2.71, -3.49, -2.57,  5.87,  8.14],
    [-4.59,  5.05,  6.63, -3.76, -1.34,  0.47, -0.86, -5.85,  1.37, -9.02],
    [-5.45,  6.20, -7.75, -7.28,  7.16,  0.84, -5.54, -6.42, -3.17, -0.69],
    [+3.04,  3.94,  3.83, -4.09,  1.80,  0.70,  2.61,  5.30,  5.66,  2.82],
    [-7.66,  2.54, -0.01,  6.51, -8.71,  0.91, -5.09,  3.63,  9.23, -2.76]
])


ME = FixedMatrix([
    [-8.93, -8.36, -8.05, -4.74, -3.79, -6.32,  1.17, -0.75, -7.12, -1.23],
    [-6.96,  2.46, -7.24,  7.27, -3.38, -4.95, -7.57,  2.18, -8.42, -7.23],
    [+5.06,  9.26, -2.75, -5.33, -4.60,  6.00,  4.92,  2.09,  9.28,  9.96],
    [-6.76, -9.33, -5.38, -5.48,  2.09,  0.10, -8.08,  9.23,  2.15, -6.41],
    [+6.81, -0.95,  2.53, -3.06,  1.98,  3.46,  5.80,  1.78,  5.75,  6.68],
    [-1.55, -4.18,  5.38,  8.89,  0.37, -0.02, -7.14, -6.30,  8.64, -0.21],
    [+8.48,  5.19, -8.43,  7.76,  2.41,  3.33,  2.67, -0.60, -9.39,  0.15],
    [+3.83, -0.73,  2.07, -9.59,  9.01, -0.87, -8.18, -5.32, -8.95,  1.68],
    [-3.58,  8.83, -6.80,  3.43,  7.87, -9.59, -8.48,  9.23, -3.22,  4.52],
    [+8.05, -2.01,  7.68,  6.67, -0.24, -7.72,  4.72,  2.07,  5.97, -7.72]
])
