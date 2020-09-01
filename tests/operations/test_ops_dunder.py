from dataclasses import dataclass

from myia.lib import ANYTHING, InferenceError
from myia.testing.common import af32_of, ai64_of, f64, i64
from myia.testing.multitest import mt

from ..test_infer import infer_standard


@dataclass
class Foo:
    x: int
    y: int

    def __pow__(self, other):
        if isinstance(other, Foo):
            return 1
        else:
            return NotImplemented

    def __rpow__(self, other):
        return 2

    def __lt__(self, other):
        return 10

    def __le__(self, other):
        return 20

    def __gt__(self, other):
        return 30

    def __ge__(self, other):
        return 40


@dataclass
class Doo:
    a: int
    b: int

    def __pow__(self, other):
        if isinstance(other, Doo):
            return 3
        else:
            return NotImplemented

    def __rpow__(self, other):
        return 4


@mt(
    infer_standard(Foo(i64, i64), Foo(i64, i64), result=1),
    infer_standard(Foo(i64, i64), Doo(i64, i64), result=4),
    infer_standard(Doo(i64, i64), Foo(i64, i64), result=2),
    infer_standard(Doo(i64, i64), Doo(i64, i64), result=3),
    infer_standard(i64, i64, result=i64),
    infer_standard(i64, ai64_of(4, 5), result=ai64_of(4, 5)),
    infer_standard(ai64_of(4, 5), i64, result=ai64_of(4, 5)),
)
def test_dunder_pow(x, y):
    return x ** y


@mt(
    infer_standard(Foo(i64, i64), Foo(i64, i64), result=(10, 20, 30, 40)),
    infer_standard(Doo(i64, i64), Foo(i64, i64), result=(30, 40, 10, 20)),
    infer_standard(Doo(i64, i64), Doo(i64, i64), result=InferenceError),
)
def test_dunder_comparisons(x, y):
    # This only tests that the protocol is correct
    return x < y, x <= y, x > y, x >= y


@mt(
    infer_standard(i64, i64, result=i64),
    infer_standard(ai64_of(7, 9), ai64_of(7, 9), result=ai64_of(7, 9)),
    infer_standard(ai64_of(7, 9), i64, result=ai64_of(7, 9)),
    infer_standard(i64, ai64_of(7, 9), result=ai64_of(7, 9)),
    infer_standard(ai64_of(7, 9), i64, result=ai64_of(7, 9)),
    infer_standard(i64, f64, result=InferenceError),
    infer_standard(3, ai64_of(7, 9), result=ai64_of(7, 9)),
    infer_standard(af32_of(7, 9), af32_of(7, 1), result=af32_of(7, 9)),
    infer_standard(af32_of(1, 9), af32_of(7, 1), result=af32_of(7, 9)),
    infer_standard(
        af32_of(1, ANYTHING), af32_of(7, 1), result=af32_of(7, ANYTHING)
    ),
    infer_standard(
        af32_of(8, ANYTHING), af32_of(8, ANYTHING), result=af32_of(8, ANYTHING)
    ),
    infer_standard(af32_of(8, 3), af32_of(8, ANYTHING), result=af32_of(8, 3)),
    infer_standard(af32_of(2, 3, 4), af32_of(3, 4), result=af32_of(2, 3, 4)),
    infer_standard(ai64_of(7), ai64_of(9), result=InferenceError),
)
def test_add(x, y):
    return x + y


@mt(
    infer_standard(f64, result=f64),
    infer_standard(i64, result=i64),
    infer_standard(af32_of(2, 5), result=af32_of(2, 5)),
)
def test_add1(x):
    return 1 + x


def _add(x, y):
    return x + y


@mt(infer_standard(f64, result=f64), infer_standard(i64, result=i64))
def test_add1_indirect(x):
    return _add(1, x)


def _interference_helper(x):
    if isinstance(x, tuple):
        return x[0]
    else:
        return x


@mt(infer_standard(i64, result=i64), infer_standard(f64, result=f64))
def test_add1_hastype_interference(x):
    return x + _interference_helper(1)
