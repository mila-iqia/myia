
import numpy
# composite has to be imported before hypermap because of circular import
# shenanigans
from myia import composite  # noqa
from myia.hypermap import HyperMap, hyper_map
from myia.prim.py_implementations import scalar_add

from .common import Point


def test_hypermap():
    # Normal
    assert hyper_map(scalar_add, 10, 20) == 30
    assert hyper_map(scalar_add, (1, 2), (10, 20)) == (11, 22)
    assert hyper_map(scalar_add, [1, 2, 3], [3, 2, 1]) == [4, 4, 4]
    assert (hyper_map(scalar_add, numpy.ones((2, 2)), numpy.ones((2, 2)))
            == 2 * numpy.ones((2, 2))).all()

    # Broadcast
    assert hyper_map(scalar_add, [1, 2, 3], 10) == [11, 12, 13]
    assert hyper_map(scalar_add, Point([1, 2], (3, 4)), Point(10, 100)) \
        == Point([11, 12], (103, 104))
    assert (hyper_map(scalar_add, numpy.ones((2, 2)), 9)
            == 10 * numpy.ones((2, 2))).all()

    # Provide fn_leaf
    adder = HyperMap(fn_leaf=scalar_add)
    assert adder((1, 2), (10, 20)) == (11, 22)
    assert (adder(numpy.ones((2, 2)), numpy.ones((2, 2)))
            == 2 * numpy.ones((2, 2))).all()


def test_arithmetic_data():
    assert Point(1, 2) + Point(10, 20) == Point(11, 22)
    assert Point(1, 2) + 10 == Point(11, 12)
