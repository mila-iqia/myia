from dataclasses import dataclass

import numpy

from myia import lib
from myia.hypermap import HyperMap
from myia.lib import InferenceError
from myia.operations import hyper_map, scalar_add
from myia.testing.common import (
    MA,
    MB,
    Point,
    Point3D,
    U,
    af64_of,
    ai64_of,
    f64,
    i64,
)
from myia.testing.multitest import mt, run, run_debug
from myia.utils import tags

from ..test_algos import make_tree
from ..test_infer import infer_scalar

hyper_map_notuple = HyperMap(
    nonleaf=(lib.AbstractArray, lib.AbstractUnion, lib.AbstractClassBase)
)
hyper_map_nobroadcast = HyperMap(broadcast=False)


@mt(
    infer_scalar(i64, i64, result=i64),
    infer_scalar(f64, f64, result=f64),
    infer_scalar([f64], [f64], result=[f64]),
    infer_scalar([[f64]], [[f64]], result=[[f64]]),
    infer_scalar((i64, f64), (i64, f64), result=(i64, f64)),
    infer_scalar(Point(i64, i64), Point(i64, i64), result=Point(i64, i64)),
    infer_scalar(ai64_of(2, 5), ai64_of(2, 5), result=ai64_of(2, 5)),
    infer_scalar(ai64_of(2, 5), i64, result=ai64_of(2, 5)),
    infer_scalar(ai64_of(1, 5), ai64_of(2, 1), result=ai64_of(2, 5)),
    infer_scalar(i64, f64, result=InferenceError),
    infer_scalar(ai64_of(2, 5), af64_of(2, 5), result=InferenceError),
    infer_scalar(
        U(i64, (i64, i64)), U(i64, (i64, i64)), result=U(i64, (i64, i64))
    ),
    infer_scalar(
        U(i64, (i64, i64)), U(i64, (i64, i64, i64)), result=InferenceError
    ),
    infer_scalar(
        {"x": i64, "y": i64}, {"x": i64, "y": i64}, result={"x": i64, "y": i64}
    ),
    infer_scalar(
        {"x": i64, "y": i64},
        {"x": i64, "y": i64, "z": i64},
        result=InferenceError,
    ),
    infer_scalar(
        {"x": i64, "y": i64}, {"y": i64, "z": i64}, result=InferenceError
    ),
    # Generic broadcasting tests
    infer_scalar([f64], f64, result=[f64]),
    infer_scalar([[f64]], [[f64]], result=[[f64]]),
    infer_scalar((i64, i64), i64, result=(i64, i64)),
    infer_scalar(i64, (i64, i64), result=(i64, i64)),
    infer_scalar(Point(i64, i64), i64, result=Point(i64, i64)),
    # Various errors
    infer_scalar((i64, i64), (i64, i64, i64), result=InferenceError),
    infer_scalar(
        Point(i64, i64), Point3D(i64, i64, i64), result=InferenceError
    ),
    infer_scalar((i64, i64), [i64], result=InferenceError),
    # Full tests
    run(MA(2, 3), MB(2, 3)),
    run(Point(1, 2), Point(3, 4)),
    run((MA(2, 3), 7.5, MB(1, 3)), 3.5),
)
def test_hyper_map(x, y):
    return hyper_map(scalar_add, x, y)


@mt(
    infer_scalar((i64, f64), (i64, f64), result=InferenceError),
    infer_scalar([f64], f64, result=[f64]),
)
def test_hyper_map_notuple(x, y):
    return hyper_map_notuple(scalar_add, x, y)


@mt(
    infer_scalar(ai64_of(2, 5), ai64_of(2, 5), result=ai64_of(2, 5)),
    infer_scalar(ai64_of(2, 5), ai64_of(2, 1), result=InferenceError),
    infer_scalar(ai64_of(2, 5), i64, result=InferenceError),
)
def test_hyper_map_nobroadcast(x, y):
    return hyper_map_nobroadcast(scalar_add, x, y)


@run((1.5, 2.6, 3.7))
def test_hyper_map_ct(x):
    return hyper_map(scalar_add, x, 1)


@run_debug(make_tree(3, 1), validate=False)
def test_hypermap_tree(t):
    return hyper_map(scalar_add, t, t)


###############################
# Test Python implementations #
###############################


def test_hypermap_python():
    # Normal
    assert hyper_map(scalar_add, 10, 20) == 30
    assert hyper_map(scalar_add, (1, 2), (10, 20)) == (11, 22)
    assert hyper_map(scalar_add, [1, 2, 3], [3, 2, 1]) == [4, 4, 4]
    assert (
        hyper_map(scalar_add, numpy.ones((2, 2)), numpy.ones((2, 2)))
        == 2 * numpy.ones((2, 2))
    ).all()

    # Broadcast
    assert hyper_map(scalar_add, [1, 2, 3], 10) == [11, 12, 13]
    assert hyper_map(
        scalar_add, Point([1, 2], (3, 4)), Point(10, 100)
    ) == Point([11, 12], (103, 104))
    assert (
        hyper_map(scalar_add, numpy.ones((2, 2)), 9) == 10 * numpy.ones((2, 2))
    ).all()

    # Provide fn_leaf
    adder = HyperMap(fn_leaf=scalar_add)
    assert adder((1, 2), (10, 20)) == (11, 22)
    assert (
        adder(numpy.ones((2, 2)), numpy.ones((2, 2))) == 2 * numpy.ones((2, 2))
    ).all()


def test_arithmetic_data_python():
    assert Point(1, 2) + Point(10, 20) == Point(11, 22)
    assert Point(1, 2) + 10 == Point(11, 12)


@dataclass
class ConstYPoint:
    x: tags.Update
    y: object


add_update = HyperMap(fn_leaf=scalar_add, update_tag=tags.Update)


@mt(
    run(ConstYPoint(5, 10), result=ConstYPoint(6, 10)),
    run((3, ConstYPoint(5, 10)), result=(4, ConstYPoint(6, 10))),
    run(
        ConstYPoint(ConstYPoint(10, 20), ConstYPoint(30, 40)),
        result=ConstYPoint(ConstYPoint(11, 20), ConstYPoint(30, 40)),
    ),
)
def test_partial_update(pt):
    return add_update(pt, 1)


@run(ConstYPoint(5, 10), ConstYPoint(5, 10), result=ConstYPoint(10, 10))
def test_partial_update_binary(pt1, pt2):
    return add_update(pt1, pt2)


@run(ConstYPoint(5, 10), result=InferenceError)
def test_bad_update(pt):
    return add_update(1, pt)


def test_partial_update_python():
    pt = ConstYPoint(5, 10)
    assert add_update(pt, 1) == ConstYPoint(6, 10)
