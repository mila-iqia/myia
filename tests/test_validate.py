
import pytest

from myia.pipeline import scalar_pipeline, scalar_parse
from myia.composite import list_map
from myia.dtype import Array, Tuple, List, Int
from myia.dshape import NOSHAPE, TupleShape, ListShape
from myia.prim import ops as P
from myia.prim.py_implementations import make_record, partial
from myia.validate import validate as _validate, ValidationError, \
    _validate_shape

from .common import L, i64, Point, Point_t


test_whitelist = frozenset({
    P.scalar_add,
    P.scalar_sub,
    P.scalar_mul,
    P.scalar_div,
    P.make_tuple,
    P.tuple_getitem,
    # P.list_map,
    P.bool_and,
    P.scalar_lt,
    P.list_len,
    P.list_append,
    P.list_getitem,
    P.make_list,
    P.partial,
    P.switch,
    P.return_,
})


def validate(g):
    return _validate(g, whitelist=test_whitelist)


pip = scalar_pipeline \
    .select('parse', 'infer', 'specialize', 'validate') \
    .configure({'validate.whitelist': test_whitelist})


pip_ec = scalar_pipeline \
    .select('parse', 'infer', 'specialize',
            'erase_class', 'erase_tuple', 'validate') \
    .configure({'validate.whitelist': test_whitelist})


def run(pip, fn, types):
    res = pip.run(input=fn, argspec=[{'type': t} for t in types])
    return res['graph']


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


def test_validate_shape():
    # Can't cover this if the type/shape inferrers do their job correctly.

    with pytest.raises(ValidationError):
        _validate_shape(List[Int[64]], (1, 2, 3))

    with pytest.raises(ValidationError):
        _validate_shape(Array[Int[64]],
                        ListShape((1, 2, 3)))

    with pytest.raises(ValidationError):
        _validate_shape(List[Array[Int[64]]],
                        ListShape(ListShape((1, 2, 3))))

    with pytest.raises(ValidationError):
        _validate_shape(Tuple[Array[Int[64]]],
                        ListShape((1, 2, 3)))

    with pytest.raises(ValidationError):
        _validate_shape(Tuple[Int[64], Int[64], Int[64]],
                        TupleShape([NOSHAPE, NOSHAPE]))
