
from dataclasses import dataclass

from ..common import ai64_of, i64
from ..multitest import mt
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
def test_dunder(x, y):
    return x ** y


# class Z:
#     def __eq__(self, other):
#         print(other)
#         return NotImplemented

#     def __ne__(self, other):
#         return NotImplemented


# class Q:
#     def __eq__(self, other):
#         return False

#     def __ne__(self, other):
#         return False


# def test_stuff():
#     z1 = Z()
#     z2 = Z()
#     q1 = Q()
#     q2 = Q()
#     assert z1 == z2
