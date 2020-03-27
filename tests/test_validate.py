import pytest

from myia.abstract import AbstractFunction, VirtualFunction
from myia.operations import partial, primitives as P
from myia.pipeline import scalar_parse, scalar_pipeline
from myia.validate import ValidationError, validate, validate_abstract

from .common import Point, f64, i64, to_abstract_test

Point_a = Point(i64, i64)


pip = scalar_pipeline.select(
    "resources", "parse", "infer", "specialize", "validate"
)


pip_ec = scalar_pipeline.select(
    "resources", "parse", "infer", "specialize", "simplify_types", "validate"
)


def run(pip, fn, types):
    res = pip.run(input=fn, argspec=[to_abstract_test(t) for t in types])
    return res["graph"]


def valid(*types):
    def deco(fn):
        run(pip, fn, types)

    return deco


def invalid(*types):
    def deco(fn):
        with pytest.raises(ValidationError):
            run(pip, fn, types)

    return deco


def valid_after_ec(*types):
    def deco(fn):
        invalid(*types)(fn)
        run(pip_ec, fn, types)

    return deco


def test_validate():
    @valid(i64, i64)
    def f1(x, y):
        return x + y

    # make_record is not in the whitelist
    @invalid(i64, i64)
    def f2(x, y):
        return Point(x, y)

    # make_record is not in the whitelist
    # simplify_types should remove it
    @valid_after_ec(i64, i64)
    def f3(x, y):
        return Point(x, y)

    @valid()
    def f4():
        return 123

    # Cannot have String types
    @invalid()
    def f5():
        return "hello"

    @valid(i64, i64)
    def f6(x, y):
        return x.__add__(y)

    # Cannot have Class types
    # simplify_types should remove it
    @valid_after_ec(Point_a)
    def f7(pt):
        return pt

    # Cannot have getattr
    # simplify_types should remove it
    @valid_after_ec(Point_a)
    def f8(pt):
        return pt.x

    @scalar_parse
    def f9(x):
        return x

    with pytest.raises(ValidationError):
        validate(f9)


def test_clean():
    @valid_after_ec(i64, i64)
    def f1(x, y):
        return P.make_record(Point, x, y)

    @valid_after_ec(i64, i64)
    def f2(x, y):
        return partial(P.make_record, Point, x)(y)

    @valid_after_ec((Point_a, Point_a))
    def f3(xs):
        def f(pt):
            return pt.x + pt.y

        return f(xs[0]), f(xs[1])

    @valid_after_ec(i64, i64)
    def f4(x, y):
        def f():
            return partial(P.make_record, Point)

        return f()(x, y)

    @valid_after_ec(i64, i64)
    def f5(x, y):
        def f(x):
            return partial(P.make_record, Point, x)

        return f(x)(y)


def test_validate_abstract():
    fn = AbstractFunction(
        VirtualFunction((), to_abstract_test(i64)),
        VirtualFunction((), to_abstract_test(f64)),
    )
    with pytest.raises(ValidationError):
        validate_abstract(fn, {})
