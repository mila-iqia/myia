
import pytest

from myia.api import scalar_pipeline, scalar_parse
from myia.prim import ops as P
from myia.prim.py_implementations import make_record, partial, list_map
from myia.validate import validate as _validate, ValidationError
from myia.opt.clean import erase_class

from .common import L, i64, Point, Point_t


test_whitelist = frozenset({
    P.scalar_add,
    P.scalar_sub,
    P.scalar_mul,
    P.scalar_div,
    P.make_tuple,
    P.tuple_getitem,
    P.list_map,
    P.partial,
    P.if_,
    P.return_,
})


def validate(g):
    return _validate(g, whitelist=test_whitelist)


pip = scalar_pipeline \
    .select('parse', 'infer', 'specialize')


def make(*types):
    def deco(fn):
        res = pip.run(input=fn, argspec=[{'type': t} for t in types])
        return res['graph']
    return deco


def valid(*types):
    def deco(fn):
        g = make(*types)(fn)
        validate(g)
    return deco


def invalid(*types):
    def deco(fn):
        g = make(*types)(fn)
        with pytest.raises(ValidationError):
            validate(g)
    return deco


def valid_after_ec(*types):
    def deco(fn):
        g = make(*types)(fn)
        with pytest.raises(ValidationError):
            validate(g)
        erase_class(g, g.manager)
        validate(g)
    return deco


def test_validate():
    @valid(i64, i64)
    def f1(x, y):
        return x + y

    # ** is not in the whitelist
    @invalid(i64, i64)
    def f2(x, y):
        return x ** y

    # make_record is not in the whitelist
    # erase_class should remove it
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
    # erase_class should remove it
    @valid_after_ec(Point_t)
    def f7(pt):
        return pt

    # Cannot have getattr
    # erase_class should remove it
    @valid_after_ec(Point_t)
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
        return make_record(Point_t, x, y)

    @valid_after_ec(i64, i64)
    def f2(x, y):
        return partial(make_record, Point_t, x)(y)

    @valid_after_ec(L[Point_t])
    def f3(xs):
        def f(pt):
            return pt.x + pt.y
        return list_map(f, xs)
