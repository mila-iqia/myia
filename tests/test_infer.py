
import pytest
import operator
import numpy as np

import typing
from types import SimpleNamespace

from myia import abstract
from myia.abstract import concretize_abstract
from myia.abstract.prim import UniformPrimitiveInferrer
from myia.pipeline import standard_pipeline, scalar_pipeline
from myia.composite import hyper_add, zeros_like, grad, list_map, tail
from myia.debug.traceback import print_inference_error
from myia.dtype import Int, External, List, Number, Class, EnvType as Env, Nil
from myia.hypermap import HyperMap, hyper_map
from myia.abstract import ANYTHING, InferenceError, Contextless, CONTEXTLESS
from myia.ir import Graph, MultitypeGraph
from myia.prim import Primitive, ops as P
from myia.prim.py_implementations import \
    scalar_add, scalar_mul, scalar_lt, list_map as list_map_prim, \
    hastype, typeof, scalar_usub, dot, distribute, shape, array_map, \
    array_reduce, reshape, partial as myia_partial, identity, \
    bool_and, bool_or, switch, scalar_to_array, broadcast_shape, \
    tuple_setitem, list_setitem, scalar_cast, list_reduce, \
    env_getitem, env_setitem, embed, J, Jinv, array_to_scalar, \
    transpose, make_record, unsafe_static_cast
from myia.utils import newenv

from .common import B, T, L, i16, i32, i64, u64, f16, f32, f64, \
    ai64, af64, Point, Point_t, Point3D, Thing, Thing_ftup, mysum, \
    ai64_of, ai32_of, af64_of, af32_of, af16_of, S, Ty, JT, Shp, \
    to_abstract_test, EmptyTuple, U


########################
# Temporary primitives #
########################


pyimpl_test = {}
abstract_inferrer_cons_test = {}


def _test_op(fn):
    prim = Primitive(fn.__name__)
    xinf = UniformPrimitiveInferrer.partial(prim=prim, impl=fn)
    abstract_inferrer_cons_test[prim] = xinf
    return prim


@_test_op
def _tern(x: Number, y: Number, z: Number) -> Number:
    return x + y + z


@_test_op
def _to_i64(x: Number) -> Int[64]:
    return int(x)


infer_pipeline = scalar_pipeline.select(
    'parse', 'infer'
).configure({
    'py_implementations': pyimpl_test,
    'inferrer.constructors': abstract_inferrer_cons_test,
})


infer_pipeline_std = standard_pipeline.select(
    'parse', 'infer'
).configure({
    'py_implementations': pyimpl_test,
    'inferrer.constructors': abstract_inferrer_cons_test,
})


def _is_exc_type(cls):
    return isinstance(cls, type) and issubclass(cls, Exception)


def inferrer_decorator(pipeline):
    def infer(*tests):

        tests = [[to_abstract_test(x) for x in test] for test in tests]

        def decorate(fn):
            def run_test(spec):
                *args, expected_out = spec

                print('Args:')
                print(args)

                def out():
                    pip = pipeline.make()
                    res = pip(input=fn, argspec=args)
                    rval = res['outspec']

                    print('Output of inferrer:')
                    rval = pip.resources.inferrer.engine.run_coroutine(
                        concretize_abstract(rval)
                    )
                    print(rval)
                    return rval

                print('Expected:')
                print(expected_out)

                if _is_exc_type(expected_out):
                    try:
                        out()
                    except expected_out as e:
                        if issubclass(expected_out, InferenceError):
                            print_inference_error(e)
                        else:
                            pass
                    else:
                        raise Exception(
                            f'Expected {expected_out}, got: (see stdout).'
                        )
                else:
                    try:
                        assert out() == expected_out
                    except InferenceError as e:
                        print_inference_error(e)
                        raise

            m = pytest.mark.parametrize('spec', list(tests))(run_test)
            m.__orig__ = fn
            return m

        return decorate

    return infer


infer = inferrer_decorator(infer_pipeline)
infer_std = inferrer_decorator(infer_pipeline_std)


type_signature_arith_bin = [
    (i64, i64, i64),
    (f64, f64, f64),
    (i64, f64, InferenceError),
    (B, B, InferenceError),
]


def test_contextless():
    C = CONTEXTLESS
    assert Contextless.empty() is C
    assert C.filter(Graph()) is C
    assert C.add(Graph(), []) is C


@infer((i64, i64),
       (89, 89))
def test_identity(x):
    return x


@infer((i64,))
def test_constants_int():
    return 2 * 8


@infer((f64,))
def test_constants_float():
    return 1.5 * 8.0


@infer((f64,))
def test_constants_intxfloat():
    return 8 * 1.5


@infer((f64,))
def test_constants_floatxint():
    return 1.5 * 8


@infer((f64,))
def test_constants_floatxint2():
    return (8 * 7) + 4.0


@infer(*type_signature_arith_bin)
def test_prim_mul(x, y):
    return x * y


@infer(
    (i64, i64, i64, i64),
    (f64, f64, f64, f64),
    # Three different inconsistent patterns below
    (f64, f64, i64, InferenceError),
    (i64, f64, f64, InferenceError),
    (f64, f64, i64, InferenceError),
    # Test too few/too many arguments below
    (i64, InferenceError),
    (i64, i64, i64, i64, InferenceError),
)
def test_prim_tern(x, y, z):
    return _tern(x, y, z)


@infer((i64, i64), (f64, f64), (B, InferenceError))
def test_prim_usub(x):
    return -x


@infer_std(
    (i64, InferenceError),
    (f32, f32),
    (f64, f64),
    (af64_of(2, 5), af64_of(2, 5)),
    (B, InferenceError)
)
def test_prim_log(x):
    return np.log(x)


@infer(
    (B, f64, f64, f64),
    (B, f64, i64, InferenceError),
    (True, f64, i64, f64),
    (False, f64, i64, i64),
    # Note: scalar_pipeline will not convert i64 to bool,
    # so the following is an InferenceError even though it
    # will work with the standard_pipeline
    (i64, f64, f64, InferenceError),
    (True, 7, 4, i64),
    (False, 7, 4, i64),
    (B, 7, 4, i64),
)
def test_if(c, x, y):
    if c:
        return x * x
    else:
        return y * y


@infer(*type_signature_arith_bin)
def test_if2(x, y):
    if x > y:
        return x
    else:
        return y


@infer(
    (i64, i64, i64),
    (i64, f64, f64),
    (f64, f64, f64),
    (1_000_000, 3, i64)
)
def test_while(x, y):
    rval = y
    while x > 0:
        rval = rval * y
        x = x - 1
    return rval


@infer(
    ([i64], i64, i64),
    ([i64], f64, InferenceError),
    (i64, i64, InferenceError),
    ((i64, i64, i64), i64, i64),
    ((i64, f64, i64), i64, InferenceError),
)
def test_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


@infer((i64, f64, (i64, f64)))
def test_nullary_closure(x, y):
    def make(z):
        def inner():
            return z
        return inner
    a = make(x)
    b = make(y)
    return a(), b()


@infer((i64, f64, (i64, f64)))
def test_merge_point(x, y):
    def mul2():
        return scalar_mul
    m = mul2()
    return m(x, x), m(y, y)


@infer((i64, InferenceError))
def test_not_enough_args_prim(x):
    return scalar_mul(x)


@infer((i64, i64, i64, InferenceError))
def test_too_many_args_prim(x, y, z):
    return scalar_mul(x, y, z)


@infer((i64, InferenceError))
def test_not_enough_args(x):
    def g(x, y):
        return x * y
    return g(x)


@infer((i64, i64, InferenceError))
def test_too_many_args(x, y):
    def g(x):
        return x * x
    return g(x, y)


@infer((i64, f64, (i64, f64)),
       ((i64, i64), f64, ((i64, i64), f64)))
def test_tup(x, y):
    return (x, y)


@infer((i64, i64, [i64]),
       (i64, f64, InferenceError),
       ([ai64_of(8, 3)], [ai64_of(4, 3)], [[ai64_of(ANYTHING, 3)]]),
       (ai64_of(4, 7), ai64_of(4, 7), [ai64_of(4, 7)]),
       (ai64_of(4, 7), ai64_of(9, 7), [ai64_of(ANYTHING, 7)]))
def test_list(x, y):
    return [x, y]


@infer((i64, i64, [i64]),
       (f64, f64, [f64]),
       ([f64], [f64], InferenceError),
       (i64, f64, InferenceError))
def test_list_and_scalar(x, y):
    return [x, y, 3]


# @infer(type=[(L[Problem[VOID]],)],
#        shape=[(InferenceError,)])
# def test_list_empty():
#     return []


@infer(
    ((), 0),
    ((1,), 1),
    ((i64, f64), 2),
    ([f64], InferenceError),
    (af64_of(2, 5), InferenceError),
    (i64, InferenceError),
)
def test_tuple_len(xs):
    return P.tuple_len(xs)


@infer(
    ((i64, f64), InferenceError),
    ([f64], i64),
    (af64_of(2, 5), InferenceError),
    (i64, InferenceError),
)
def test_list_len(xs):
    return P.list_len(xs)


@infer(
    ((i64, f64), InferenceError),
    ([f64], InferenceError),
    (af64_of(2, 5), i64),
    (i64, InferenceError),
)
def test_array_len(xs):
    return P.array_len(xs)


@infer(
    ((i64, f64), (f64,)),
    ((f64, i64), (i64,)),
    ((f64, (i64, f64)), ((i64, f64),)),
    ((), InferenceError),
    (f64, InferenceError),
)
def test_tail_tuple(tup):
    return tail(tup)


@infer(((i64), (f64,), InferenceError))
def test_tail_tuple_wrong(x, y):
    return tail(x, y)


@infer((i64, f64, i64), (f64, i64, f64))
def test_tuple_getitem(x, y):
    return (x, y)[0]


@infer((i64, f64, f64), (f64, i64, i64))
def test_tuple_getitem_negative(x, y):
    return (x, y)[-1]


@infer((i64, f64, InferenceError))
def test_tuple_outofbound(x, y):
    return (x, y)[2]


@infer((i64, f64, InferenceError))
def test_tuple_outofbound_negative(x, y):
    return (x, y)[-3]


@infer(
    ([i64], i64, i64),
    ([f64], i64, f64),
    ([f64], f64, InferenceError),
    (f64, i64, InferenceError),
    ((i64, f64), i64, InferenceError)
)
def test_list_getitem(xs, i):
    return xs[i]


@infer(
    ((i64, i64), 1, f64, (i64, f64)),
    ((i64, i64, f64), 1, f64, (i64, f64, f64)),
    ((i64,), 1, f64, InferenceError),
    ((i64,), 0.0, f64, InferenceError),
    ((i64,), i64, f64, InferenceError),
)
def test_tuple_setitem(xs, idx, x):
    return tuple_setitem(xs, idx, x)


@infer(
    ([i64], i64, i64, [i64]),
    ([i64], f64, i64, InferenceError),
    ([i64], i64, f64, InferenceError),
)
def test_list_setitem(xs, idx, x):
    return list_setitem(xs, idx, x)


@infer((i64, f64, (i64, f64)))
def test_multitype_function(x, y):
    def mul(a, b):
        return a * b
    return (mul(x, x), mul(y, y))


@infer(*type_signature_arith_bin)
def test_closure(x, y):
    def mul(a):
        return a * x
    return mul(x) + mul(y)


@infer(
    (i64, i64, i64, i64, (i64, i64)),
    (f64, f64, f64, f64, (f64, f64)),
    (i64, i64, f64, f64, (i64, f64)),
    (i64, f64, f64, f64, InferenceError),
    (i64, i64, i64, f64, InferenceError),
)
def test_return_closure(w, x, y, z):
    def mul(a):
        def clos(b):
            return a * b
        return clos
    return (mul(w)(x), mul(y)(z))


@infer(
    (i64, i64),
    (f64, f64),
)
def test_fact(n):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(n)


def even(n):
    if n == 0:
        return True
    else:
        return odd(n - 1)


def odd(n):
    if n == 0:
        return False
    else:
        return even(n - 1)


@infer((i64, B), (f64, B))
def test_even_odd(n):
    return even(n)


@infer((i64, i64), (f64, f64))
def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v


@infer(
    (i64, i64, i64, i64),
    (i64, f64, f64, f64)
)
def test_choose_prim(i, x, y):

    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_mul

    return choose(i)(x, y)


@infer(
    (i64, i64, i64, InferenceError),
    (0, i64, i64, i64),
    (1, i64, i64, B),
)
def test_choose_prim_incompatible(i, x, y):

    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_lt

    return choose(i)(x, y)


@infer(
    (i64, i64, i64, InferenceError),
    (0, i64, i64, i64),
    (1, i64, i64, B),
)
def test_choose_incompatible(i, x, y):

    def add2(x, y):
        return x + y

    def lt2(x, y):
        return x < y

    def choose(i):
        if i == 0:
            return add2
        else:
            return lt2

    return choose(i)(x, y)


@infer(
    (i64, i64, i64),
    (i64, f64, f64)
)
def test_choose_indirect(i, x):

    def double(x):
        return x + x

    def square(x):
        return x * x

    def choose(i):
        if i == 0:
            return double
        else:
            return square

    return choose(i)(x)


@infer((i64, i64))
def test_hof(x):

    def double(x):
        return x + x

    def square(x):
        return x * x

    def hof(f, tup):
        return f(tup[0]) + f(tup[1])

    return hof(double, (x + 1, x + 2)) + hof(square, (x + 3, x + 4))


@infer(
    (i64, i64, i64),
    (i64, f64, InferenceError),
    (i64, 3, i64),
)
def test_hof_2(c, x):

    def identity(x):
        return x

    def double(x):
        return x + x

    def square(x):
        return x * x

    def pick(c):
        if c < 0:
            return square
        elif c == 0:
            return _to_i64
        else:
            return double

    def pick2(c, f):
        if c < 0:
            return f
        else:
            return identity

    return pick2(c, pick(c))(x + x)


@infer((i64, ((i64, i64), (B, B))))
def test_hof_3(x):

    def double(x):
        return x + x

    def is_zero(x):
        return x == 0

    def hof(f, tup):
        return (f(tup[0]), f(tup[1]))

    return (hof(double, (x + 1, x + 2)), hof(is_zero, (x + 3, x + 4)))


@infer(
    (i64, i64, InferenceError),
    (-1, i64, i64),
    (1, i64, (i64, i64)),
)
def test_hof_4(x, y):

    def double(x):
        return x + x

    def hof_sum(f, tup):
        return f(tup[0]) + f(tup[1])

    def hof_tup(f, tup):
        return (f(tup[0]), f(tup[1]))

    def pick(x):
        if x < 0:
            return hof_sum
        else:
            return hof_tup

    f = pick(x)
    return f(double, (y + 3, y + 4))


@infer(
    (B, B, i64, i64, i64),
    (B, B, f64, f64, InferenceError),
    (True, B, (), i64, i64),
    (B, True, f64, f64, f64),
    (B, True, i64, f64, InferenceError),
)
def test_hof_5(c1, c2, x, y):

    def pick_hof(c):
        def hof_1(f):
            def wrap(x, y):
                return f(y)
            return wrap

        def hof_2(f):
            def wrap(x, y):
                return f(x)
            return wrap

        if c:
            return hof_1
        else:
            return hof_2

    def pick_f(c):
        if c:
            return scalar_usub
        else:
            return _to_i64

    return pick_hof(c1)(pick_f(c2))(x, y)


@infer((i64, i64, i64))
def test_func_arg(x, y):
    def g(func, x, y):
        return func(x, y)

    def h(x, y):
        return x + y
    return g(h, x, y)


@infer((i64, InferenceError))
def test_func_arg3(x):
    def g(func, x):
        z = func + x
        return func(z)

    def h(x):
        return x

    return g(h, x)


@infer(
    (i64, i64),
    (f64, f64),
)
def test_func_arg4(x):
    def h(x):
        return x

    def g(fn, x):
        return fn(h, x)

    def t(fn, x):
        return fn(x)

    return g(t, x)


@infer((i64,))
def test_closure_deep():
    def g(x):
        def h():
            return x * x
        return h
    return g(2)()


@infer(
    (i64, i64, i64),
)
def test_closure_passing(x, y):
    def adder(x):
        def f(y):
            return x + y
        return f

    a1 = adder(1)
    a2 = adder(2)

    return a1(x) + a2(y)


@infer((B, B), (i64, InferenceError))
def test_not(x):
    return not x


@infer(
    ([i64], [f64], ([i64], [f64])),
    ([i64], f64, InferenceError),
)
def test_list_map_prim(xs, ys):

    def square(x):
        return x * x

    return list_map_prim(square, xs), list_map_prim(square, ys)


@infer(
    ([i64], [i64], [i64]),
    ([i64], [f64], InferenceError),
)
def test_list_map_prim2(xs, ys):

    def mulm(x, y):
        return x * -y

    return list_map_prim(mulm, xs, ys)


@infer(
    (i64, True),
    (f64, False),
)
def test_hastype_simple(x):
    return hastype(x, i64)


@infer(
    (i64, i64, InferenceError),
    (i64, Ty(i64), True),
    (f64, Ty(i64), False),
    ((i64, i64), Ty(T[i64, i64]), True),
    ((i64, i64), Ty(T[Number, Number]), True),
    (Point(1, 2), Ty(Class), True),
    ([i64], Ty(List[i64]), True),
    (None, Ty(Nil), True),
    ((i64, i64), Ty(ANYTHING), InferenceError),
    ((i64, i64), Ty(typing.Tuple), True),
    # ((i64, i64), Ty(tuple), True),
    # ((i64, i64), Ty(typing.Tuple[int, int]), True),
    # ((i64, i64), Ty(typing.Tuple[float, float]), False),
)
def test_hastype(x, y):
    return hastype(x, y)


@infer(
    # (i64, Ty(i64)),
    # (f64, Ty(f64)),
    (i64, Ty(to_abstract_test(i64))),
    (f64, Ty(to_abstract_test(f64))),
)
def test_typeof(x):
    return typeof(x)


Tf4 = T[f64, f64, f64, f64]


@infer(
    (i64, i64),
    (f64, i64),
    (ai64_of(2, 5), 0.0),
    (af64_of(2, 5), 0),
    ((i64, i64), i64),
    ((f64, f64, i64, i64), i64),
    ((f64, f64, f64, f64), (f64, f64, f64, f64)),
    ((i64, (f64, i64)), i64),
    ([i64], 1.0),
    ((i64, [i64]), i64),
    (Point(i64, i64), i64),
    (Point3D(i64, i64, i64), 0),
    # TODO
    # (Thing_ftup, T[f64, f64]),
    # (Thing_f, i64),

    (5, 5),
    (Point3D(5, 7, 9), 0),
)
def test_hastype_2(x):

    def f(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return f(_to_i64(x))
        elif hastype(x, ai64):
            return 0.0
        elif hastype(x, Point_t):
            return f(x.x) * f(x.y)
        elif hastype(x, EmptyTuple):
            return 0
        elif hastype(x, Tf4):
            return x
        elif hastype(x, T):
            return f(x[0]) + f(tail(x))
        elif hastype(x, L):
            return 1.0
        elif hastype(x, Thing_ftup):
            return x.contents
        else:
            return 0

    return f(x)


@infer_std(
    (U(i64, (i64, i64)), i64),
    (U(i64, (f64, i64)), InferenceError),
    (U(i64, f64), InferenceError),
)
def test_union(x):
    if hastype(x, i64):
        return x
    else:
        return x[0]


@infer_std(
    (U(i64, f64, (i64, i64)), i64, i64),
    (U(i64, f64, (i64, i64)), f64, InferenceError),
    (U(i64, (i64, i64)), f64, i64),
    (f64, f64, f64),
)
def test_union_nested(x, y):
    if hastype(x, i64):
        return x
    elif hastype(x, f64):
        return y
    else:
        return x[0]


@infer_std(
    (U(i64, f64, (i64, i64)), i64),
    (U(i64, (i64, i64)), i64),
)
def test_union_nested_2(x):
    if hastype(x, i64):
        return x
    elif hastype(x, f64):
        return 1234
    else:
        return x[0]


@infer_std(
    ([i64], i64, [i64]),
    ([f64], i64, InferenceError),
    ([ai64_of(2, 3)], ai64_of(2, 3), [ai64_of(2, 3)]),
)
def test_map_2(xs, z):

    def adder(x):
        def f(y):
            return x + y
        return f

    return list_map_prim(adder(z), xs)


def _square(x):
    return x * x


@infer((InferenceError,))
def test_nonexistent_variable():
    return xxxx + yz  # noqa


class helpers:
    add = operator.add
    mul = operator.mul
    square = _square


class data:
    a25 = np.ones((2, 5))


@infer(
    (i64, i64, (i64, i64)),
    (i64, f64, InferenceError),
)
def test_getattr(x, y):
    a = helpers.add(x, y)
    b = helpers.mul(x, y)
    c = helpers.square(b)
    return a, c


@infer(
    (i64, i64, (i64, i64)),
    (i64, f64, (i64, f64)),
)
def test_getattr_multitype(x, y):
    a = helpers.add(x, x)
    b = helpers.add(y, y)
    return a, b


@infer(
    (af64_of(2, 5),)
)
def test_getattr_shape():
    return data.a25


_getattr = getattr


@infer(
    ('add', i64, i64),
    ('bad', i64, InferenceError),
    (1234, i64, InferenceError),
    (External[str], i64, InferenceError),
)
def test_getattr_flex(name, x):
    return _getattr(helpers, name)(x, x)


@infer(
    (External[SimpleNamespace],
     S('surprise'),
     InferenceError)
)
def test_unknown_data(data, field):
    return _getattr(data, field)


@infer((i64, i64, i64), (f64, f64, f64))
def test_method(x, y):
    return x.__add__(y)


@infer((i64, i64, InferenceError))
def test_unknown_method(x, y):
    return x.unknown(y)


@infer((i64, InferenceError))
def test_infinite_recursion(x):
    def ouroboros(x):
        return ouroboros(x - 1)

    return ouroboros(x)


@infer((i64, InferenceError))
def test_indirect_infinite_recursion(x):
    def ouroboros(x):
        if x < 0:
            return ouroboros(x - 1)
        else:
            return ouroboros(x + 1)

    return ouroboros(x)


def ping():
    return pong()


def pong():
    return ping()


@infer((i64, InferenceError))
def test_infinite_mutual_recursion(x):
    return ping()


@infer(
    (af16_of(2, 3), Shp(2, 3)),
    (af16_of(2, ANYTHING), (S(2, u64), u64)),
)
def test_shape(ary):
    return shape(ary)


@infer(
    (af64_of(2, 3), af64_of(3, 4), af64_of(2, 4)),
    (af64_of(2, 3), af32_of(3, 4), InferenceError),
    (af64_of(2), af64_of(3, 4), InferenceError),
    (af64_of(2, 2), af64_of(3, 4), InferenceError),
)
def test_dot(a, b):
    return dot(a, b)


@infer(
    (ai32_of(4), Shp(2, 4), ai32_of(2, 4)),
    (ai32_of(4), (u64, u64), ai32_of(ANYTHING, ANYTHING)),
    (ai32_of(4), Shp(5, 2), InferenceError),
    (ai32_of(4, 2), Shp(4), InferenceError),
    (ai32_of(1), Shp(4), ai32_of(4)),
    (i32, Shp(4), InferenceError),
    ([i32], Shp(4), InferenceError),
)
def test_distribute(v, shp):
    return distribute(v, shp)


@infer(
    (ai32_of(3, 7), ai32_of(3, 7)),
    (ai32_of(7), ai32_of(3, 7)),
    (ai32_of(1), ai32_of(3, 7)),
    (ai32_of(1, 7), ai32_of(3, 7)),
    (ai32_of(3), InferenceError),
)
def test_distribute2(v):
    return distribute(v, (3, 7))


@infer(
    (af16_of(1, 2, 3), Shp(6), af16_of(6)),
    (af16_of(1, 2, 3), (u64,), af16_of(ANYTHING)),
    (af16_of(2, 3), Shp(7), InferenceError),
)
def test_reshape(v, shp):
    return reshape(v, shp)


@infer(
    (af16_of(6, 7), Shp(0, 1), af16_of(6, 7)),
    (af16_of(6, 7), Shp(1, 0), af16_of(7, 6)),
    (af16_of(3, 4, 5), Shp(2, 0, 1), af16_of(5, 3, 4)),
    (af16_of(3, 4, 5), (u64, u64, u64), af16_of(ANYTHING, ANYTHING, ANYTHING)),
    (af16_of(3, 4, 5), (i64, i64, i64), InferenceError),
    (af16_of(3, 4, 5), Shp(1, 0), InferenceError),
    (af16_of(3, 4, 5), Shp(1, 2, 9), InferenceError),
)
def test_transpose(v, perm):
    return transpose(v, perm)


@infer(
    (af32_of(3, 4), af32_of(3, 4)),
    (ai64_of(3, 4, 5), ai64_of(3, 4, 5)),
    (i64, InferenceError),
)
def test_array_map(ary):
    def f(v):
        return v + 1
    return array_map(f, ary)


@infer(
    (af32_of(3, 4), af32_of(3, 4), af32_of(3, 4)),
    (af32_of(3, 4), af32_of(3, 7), InferenceError),
    (ai64_of(3, ANYTHING), ai64_of(ANYTHING, 7), ai64_of(3, 7)),
    (af32_of(3, ANYTHING), af32_of(ANYTHING, ANYTHING), af32_of(3, ANYTHING)),
    (af32_of(3, 4, 5), af32_of(3, 4), InferenceError),
    (i64, af32_of(3, 4), InferenceError),
)
def test_array_map2(ary1, ary2):
    def f(v1, v2):
        return v1 + v2
    return array_map(f, ary1, ary2)


@infer((InferenceError,))
def test_array_map0():
    def f():
        return 1
    return array_map(f)


@infer(
    (af32_of(3, 4), af32_of(3, 4), af32_of(3, 4), af32_of(3, 4)),
    (af32_of(3, 4), af32_of(3, 4), af32_of(3, 7), InferenceError),
    (i64, af32_of(3, 4), af32_of(3, 4), InferenceError),
    (af32_of(3, 4), i64, af32_of(3, 4), InferenceError),
    (af32_of(3, 4), af32_of(3, 4), i64, InferenceError),
    (af32_of(3, ANYTHING, 5, 6),
     af32_of(3, 4, 5, ANYTHING),
     af32_of(ANYTHING, ANYTHING, ANYTHING, 6),
     af32_of(3, 4, 5, 6)),
    (af32_of(3, ANYTHING, 5, 6),
     af32_of(3, 4, 5, ANYTHING),
     af32_of(ANYTHING, ANYTHING, ANYTHING, 7),
     InferenceError),
    (af32_of(3, 4, 5), af32_of(3, 4), af32_of(3, 4), InferenceError),
)
def test_array_map3(ary1, ary2, ary3):
    def f(v1, v2, v3):
        return v1 + v2 + v3
    return array_map(f, ary1, ary2, ary3)


@infer(
    (ai64_of(3, 4), Shp(3, 4), ai64_of(3, 4)),
    # (ai64_of(3, 4), Shp(3, ANYTHING), ai64_of(3, ANYTHING)),
    (ai64_of(3, 4), Shp(3, 1), ai64_of(3, 1)),
    (ai64_of(3, 4), Shp(1, 4), ai64_of(1, 4)),
    (ai64_of(3, 4), Shp(3, 1, 1), InferenceError),
    (ai64_of(3, 4), Shp(4, 1), InferenceError),
    (ai64_of(3, 4), Shp(4), ai64_of(4)),
    (ai64_of(3, 4), Shp(1), ai64_of(1)),
    (ai64_of(3, 4), Shp(), ai64_of()),
    (i64, Shp(3, 4), InferenceError),
)
def test_array_reduce(ary, shp):
    def f(a, b):
        return a + b
    return array_reduce(f, ary, shp)


@infer_std(
    ([i64], i64, i64),
    ([i64], f64, InferenceError),
    ([ai64_of(6, 7)], ai64_of(6, 7), ai64_of(6, 7)),
    ([ai64_of(6, 7)], ai64_of(6, 17), InferenceError),
)
def test_list_reduce(lst, dflt):
    def f(a, b):
        return a + b
    return list_reduce(f, lst, dflt)


@infer((i64, i64))
def test_partial_1(x):
    def f(a, b):
        return a + b
    f2 = myia_partial(f, 2)
    return f2(x)


@infer((i64, i64))
def test_partial_2(x):
    def f(a, b):
        return a + b

    def g(c):
        f2 = myia_partial(f, 2)
        f3 = myia_partial(f, -2)
        if c:
            return f2
        else:
            return f3
    return g(x < 42)(x)


@infer((i64, i64), (ai64_of(6, 13), ai64_of(6, 13)))
def test_identity_function(x):
    return identity(x)


@infer((B, B, B), (i64, B, InferenceError), (B, i64, InferenceError))
def test_bool_and(x, y):
    return bool_and(x, y)


@infer((B, B, B), (i64, B, InferenceError), (B, i64, InferenceError))
def test_bool_or(x, y):
    return bool_or(x, y)


@infer(
    (B, i64, i64, i64),
    (i64, i64, i64, InferenceError),
    (B, i64, f64, InferenceError),
    (True, i64, f64, i64),
    (False, i64, f64, f64),
    (True, 1, 2, 1),
    (False, 1, 2, 2),
    (B, 1, 2, i64),
    (B, ai64_of(6, 13), ai64_of(6, 13), ai64_of(6, 13)),
    (B, ai64_of(6, 13), ai64_of(6, 14), ai64_of(6, ANYTHING)),
    (True, ai64_of(6, 13), ai64_of(6, 14), ai64_of(6, 13)),
    (False, ai64_of(6, 13), ai64_of(6, 14), ai64_of(6, 14)),
)
def test_switch(c, x, y):
    return switch(c, x, y)


@infer((i64, i64, i64))
def test_switch_switch(x, y):
    def f1(z):
        return z > 0

    def f2(z):
        return z < 0
    f = switch(x > 0, f1, f2)
    return switch(f(y), 1, 2)


@infer(
    (i64, i64, i64),
    (U(i64, f64), i64, InferenceError)
)
def test_switch_hastype(x, y):
    return switch(hastype(x, i64), y + 1, y + 2)


@infer((B, i64, i64))
def test_closure_in_data(c, x):
    def f(x):
        return x * x

    def g(x):
        return x + x

    a = Thing((1, [f]))
    b = Thing((2, [g]))
    i, h = switch(c, a, b).contents
    return h[0](x)


@infer(
    (i64, Ty(to_abstract_test(i64)), i64),
    (i64, Ty(to_abstract_test(i16)), i16),
    (f64, Ty(to_abstract_test(i16)), i16),
    (f16, Ty(to_abstract_test(f32)), f32),
    (f16, Ty(ANYTHING), InferenceError),
    (f16, Ty(to_abstract_test(B)), InferenceError),
    (B, Ty(to_abstract_test(f32)), InferenceError),
)
def test_scalar_cast(x, t):
    return scalar_cast(x, t)


@infer(
    (i64, ai64_of()),
    (f64, af64_of()),
    (af64_of(9, 7), InferenceError),
    ((i64,), InferenceError)
)
def test_scalar_to_array(x):
    return scalar_to_array(x)


@infer(
    (ai64_of(), i64),
    (af64_of(), f64),
    (af64_of(3, 4), InferenceError),
    (af64_of(1, 1, 1), InferenceError),
)
def test_array_to_scalar(x):
    return array_to_scalar(x)


@infer(
    ((u64,), (u64,), (u64,)),
    ((u64, u64), (u64,), (u64, u64)),
    ((u64,), (u64, u64), (u64, u64)),
    (Shp(1, 3), Shp(2, 1), Shp(2, 3)),
    (Shp(2, 3), Shp(2, 3), Shp(2, 3)),
    (Shp(2, 4), Shp(2, 3), InferenceError),
    ((u64, u64), Shp(2, 3), Shp(2, 3)),
    ((i64,), (u64,), InferenceError),
    ((u64,), (i64,), InferenceError),
    (i64, i64, InferenceError),
)
def test_broadcast_shape(xs, ys):
    return broadcast_shape(xs, ys)


@infer(
    (i64, i64, InferenceError)
)
def test_call_nonfunc(x, y):
    return x(y)


# @infer(type=[
#     (i64, i64, InferenceError),
#     (F[[i64], f64], i64, f64),
#     (F[[f64], f64], i64, InferenceError),
# ])
# def test_call_argument(f, x):
#     return f(x)


# @pytest.mark.xfail(reason="ExplicitInferrer generates incomplete vrefs")
# @infer(type=[
#     (F[[F[[f64], f64]], f64], f64),
# ])
# def test_call_argument_higher_order(f):
#     def g(y):
#         return y + y
#     return f(g)


@infer(
    (i64, i64, i64, i64),
    (f64, f64, f64, InferenceError),
)
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


mystery = MultitypeGraph('mystery')


@mystery.register(ai64, ai64)
def _mystery1(x, y):
    return x @ y


@mystery.register(af64, af64)
def _mystery2(x, y):
    return array_map(scalar_add, x, y)


@infer(
    (ai64_of(7, 9), ai64_of(9, 2), ai64_of(7, 2)),
    (af64_of(7, 9), af64_of(9, 2), InferenceError),
    (ai64_of(7, 9), ai64_of(7, 9), InferenceError),
    (af64_of(7, 9), af64_of(7, 9), af64_of(7, 9)),
    (f64, f64, InferenceError),
)
def test_multitype_2(x, y):
    return mystery(x, y)


# TODO: add back
# def test_forced_type():

#     @pipeline_function
#     def mod(self, graph):
#         # Force the inferred tyoe of the output to be f64
#         graph.output.inferred['type'] = f64
#         return graph

#     def fn(x, y):
#         return x + y

#     pip = infer_pipeline.insert_before('infer', mod=mod)

#     for argspec in [[{'type': i64}, {'type': i64}],
#                     [{'type': i64}, {'type': f64}]]:

#         results = pip.run(input=fn, argspec=argspec)
#         rval = results['outspec']

#         assert rval['type'] == f64


# def test_forced_function_type():

#     @pipeline_function
#     def mod(self, graph):
#         # Force the inferred tyoe of scalar_add to be (i64,i64)->f64
#         scalar_add = graph.output.inputs[0]
#         scalar_add.inferred['type'] = F[[i64, i64], f64]
#         return graph

#     def fn(x, y):
#         return x + y

#     pip = infer_pipeline.insert_before('infer', mod=mod)

#     # Test correct

#     results = pip.run(
#         input=fn,
#         argspec=[{'type': i64}, {'type': i64}]
#     )
#     rval = results['outspec']

#     assert rval['type'] == f64

#     # Test mismatch

#     with pytest.raises(InferenceError):
#         results = pip.run(
#             input=fn,
#             argspec=[{'type': i64}, {'type': f64}]
#         )

#     # Test narg mismatch

#     def fn2(x):
#         return fn(x)

#     with pytest.raises(InferenceError):
#         results = pip.run(
#             input=fn2,
#             argspec=[{'type': i64}]
#         )


# ###########################
# # Using standard_pipeline #
# ###########################


@infer_std(
    (i64, i64, i64),
    (ai64_of(7, 9), ai64_of(7, 9), ai64_of(7, 9)),
    (ai64_of(7, 9), i64, ai64_of(7, 9)),
    (i64, ai64_of(7, 9), ai64_of(7, 9)),
    (ai64_of(7, 9), i64, ai64_of(7, 9)),
    (i64, f64, InferenceError),
    (3, ai64_of(7, 9), ai64_of(7, 9)),
    (af32_of(7, 9), af32_of(7, 1), af32_of(7, 9)),
    (af32_of(1, 9), af32_of(7, 1), af32_of(7, 9)),
    (af32_of(1, ANYTHING), af32_of(7, 1), af32_of(7, ANYTHING)),
    (af32_of(8, ANYTHING), af32_of(8, ANYTHING), af32_of(8, ANYTHING)),
    (af32_of(8, 3), af32_of(8, ANYTHING), af32_of(8, 3)),
    (af32_of(2, 3, 4), af32_of(3, 4), af32_of(2, 3, 4)),
    (ai64_of(7), ai64_of(9), InferenceError),
)
def test_add_std(x, y):
    return x + y


@infer_std(
    (i64, i64, i64),
    (ai64_of(7, 9), i64, InferenceError)
)
def test_max_std(x, y):
    if x > y:
        return x
    else:
        return y


@infer_std(
    (f64, f64),
    (i64, i64),
    (af32_of(2, 5), af32_of(2, 5)),
)
def test_add1_stdx(x):
    return 1 + x


def _add(x, y):
    return x + y


@infer_std(
    (f64, f64),
    (i64, i64),
)
def test_add1_std_indirect(x):
    return _add(1, x)


def _interference_helper(x):
    if hastype(x, T):
        return x[0]
    else:
        return x


@infer(
    (i64, i64),
    (f64, f64),
)
def test_add1_hastype_interference(x):
    return x + _interference_helper(1)


# # @infer((f16, f32, f64, f32))
# @infer((f16, f32, f64, InferenceError))
# def test_hastype_interference(x, y, z):
#     if hastype(1, i32):
#         return x
#     elif hastype(1, i64):
#         return y
#     else:
#         return z


@infer(
    (Point(i64, i64), i64),
    (Point(f64, f64), f64),
)
def test_class(pt):
    return pt.x + pt.y


@infer(
    (Point(i64, i64), i64),
    (Point(f64, f64), f64),
)
def test_dataclass_method(pt):
    return pt.abs()


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_add(pt1, pt2):
    return pt1 + pt2


@infer_std((Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_add_ct(pt):
    return pt + 10


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_sub(pt1, pt2):
    return pt1 - pt2


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_mul(pt1, pt2):
    return pt1 * pt2


@infer_std((Point3D(f64, f64, f64), Point3D(f64, f64, f64),
            Point3D(f64, f64, f64)))
def test_arithmetic_data_truediv(pt1, pt2):
    return pt1 / pt2


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_floordiv(pt1, pt2):
    return pt1 // pt2


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_mod(pt1, pt2):
    return pt1 % pt2


@infer_std((Point(i64, i64), Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_pow(pt1, pt2):
    return pt1 ** pt2


@infer_std((Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_pos(pt):
    return +pt


@infer_std((Point(i64, i64), Point(i64, i64)))
def test_arithmetic_data_neg(pt):
    return -pt


@infer(
    (i64, i64, i64, i64, Point(i64, i64)),
    (f64, f64, f64, f64, InferenceError),
)
def test_dataclass_inst(x1, y1, x2, y2):
    pt1 = Point(x1, y1)
    pt2 = Point(x2, y2)
    return Point(pt1.x + pt2.x, pt1.y + pt2.y)


@infer((i64, i64, i64, InferenceError))
def test_dataclass_bad_inst(x, y, z):
    return Point(x, y, z)


@infer((Ty(ANYTHING), i64, i64, InferenceError))
def test_dataclass_bad_inst2(cls, x, y):
    return make_record(cls, x, y)


@infer((Point(i64, i64), InferenceError))
def test_dataclass_wrong_field(pt):
    return pt.z


@infer((Thing(i64), i64))
def test_dataclass_call(thing):
    return thing()


hyper_map_notuple = HyperMap(
    nonleaf=(abstract.AbstractArray,
             abstract.AbstractList,
             abstract.AbstractClass)
)
hyper_map_nobroadcast = HyperMap(broadcast=False)


@infer(
    (i64, i64, i64),
    (f64, f64, f64),
    ([f64], [f64], [f64]),
    ([[f64]], [[f64]], [[f64]]),
    ((i64, f64), (i64, f64), (i64, f64)),
    (Point(i64, i64), Point(i64, i64), Point(i64, i64)),
    (ai64_of(2, 5), ai64_of(2, 5), ai64_of(2, 5)),
    (ai64_of(2, 5), i64, ai64_of(2, 5)),
    (ai64_of(1, 5), ai64_of(2, 1), ai64_of(2, 5)),
    (i64, f64, InferenceError),
    (ai64_of(2, 5), af64_of(2, 5), InferenceError),

    # Generic broadcasting tests
    ([f64], f64, [f64]),
    ([f64], [[f64]], [[f64]]),
    ((i64, i64), i64, (i64, i64)),
    (i64, (i64, i64), (i64, i64)),
    (Point(i64, i64), i64, Point(i64, i64)),

    # Various errors
    ((i64, i64), (i64, i64, i64), InferenceError),
    (Point(i64, i64), Point3D(i64, i64, i64), InferenceError),
    ((i64, i64), [i64], InferenceError),
)
def test_hyper_map(x, y):
    return hyper_map(scalar_add, x, y)


@infer(((i64, f64), (i64, f64), InferenceError),
       ([f64], f64, [f64]))
def test_hyper_map_notuple(x, y):
    return hyper_map_notuple(scalar_add, x, y)


@infer(
    (ai64_of(2, 5), ai64_of(2, 5), ai64_of(2, 5)),
    (ai64_of(2, 5), ai64_of(2, 1), InferenceError),
    (ai64_of(2, 5), i64, InferenceError),
)
def test_hyper_map_nobroadcast(x, y):
    return hyper_map_nobroadcast(scalar_add, x, y)


@infer(
    (i64, i64, i64),
    (f64, f64, f64),
    ([f64], [f64], [f64]),
    ((i64, f64), (i64, f64), (i64, f64)),
    (Point(i64, i64), Point(i64, i64), Point(i64, i64)),
    (ai64_of(2, 5), ai64_of(2, 5), ai64_of(2, 5)),
    (ai64_of(2, 1), ai64_of(1, 5), ai64_of(2, 5)),
    (Env, Env, Env),
)
def test_hyper_add(x, y):
    return hyper_add(x, y)


@infer(
    (i64, 0),
    (f64, 0.0),
    ([f64], [f64]),  # NOTE: not sure why not [0.0] but it's not important
    ((i64, f64), (0, 0.0)),
    (Point(i64, i64), Point(0, 0)),
    (ai64_of(2, 5), ai64_of(2, 5, value=0)),
    (af32_of(2, 5), af32_of(2, 5, value=0)),
    ((2, 3.0), (0, 0.0)),
    (Point(1, 2), Point(0, 0)),
)
def test_zeros_like(x):
    return zeros_like(x)


# TODO: fix test
@infer((i64, newenv))
def test_zeros_like_fn(x):
    def f(y):
        return x + y
    return zeros_like(f)


@infer(
    ([i64], [f64], ([i64], [f64])),
    ([i64], f64, InferenceError),
)
def test_list_map(xs, ys):

    def square(x):
        return x * x

    return list_map(square, xs), list_map(square, ys)


@infer(
    ([i64], [i64], [i64]),
    ([i64], [f64], InferenceError),
)
def test_list_map2(xs, ys):

    def mulm(x, y):
        return x * -y

    return list_map(mulm, xs, ys)


@infer((InferenceError,))
def test_list_map0():

    def f():
        return 1234

    return list_map(f)


@infer(
    (i32, i32, i32, i32),
    (i32, f32, i32, InferenceError),
    (i32, i32, f32, InferenceError),
)
def test_env(x, y, z):
    e = newenv
    e = env_setitem(e, embed(x), y)
    return env_getitem(e, embed(x), z)


@infer((Env,))
def test_env_onfn():
    def f(x):
        return x * x
    e = newenv
    e = env_setitem(e, embed(f), newenv)
    return env_getitem(e, embed(f), newenv)


_i64 = to_abstract_test(i64)


@infer((i32, i64), (f64, i64), ((i32, i64), i64))
def test_unsafe_static_cast(x):
    return unsafe_static_cast(x, _i64)


@infer((i32, i32, InferenceError),
       (i32, (i32, i32), InferenceError))
def test_unsafe_static_cast_error(x, y):
    return unsafe_static_cast(x, y)


@infer(
    (i32, (JT(i32), Env, i32)),
    (f64, (JT(f64), Env, f64)),
)
def test_J(x):
    def f(x):
        return x * x

    jf = J(f)
    jx = J(x)
    jy, bprop = jf(jx)
    df, dx = bprop(1.0)
    return jy, df, dx


@infer(
    (JT(i32), i32),
    (JT([i32]), [i32]),
    (i32, InferenceError),
)
def test_Jinv(x):
    return Jinv(x)


@infer_std(
    (i32, i32),
    (f64, f64),
    (ai64_of(4, 5), ai64_of(4, 5)),
)
def test_Jinv2(x):
    def f(x):
        return x * x

    ff = Jinv(J(f))
    return ff(x)


@infer((i32, InferenceError))
def test_Jinv3(x):
    def f(x):
        return x * x
    return Jinv(f)(x)


@infer((i32, InferenceError))
def test_Jinv4(x):
    return Jinv(scalar_add)(x)


@infer_std(
    (af32_of(5, 7), (f32, (Env, af32_of(5, 7)))),
)
def test_J_array(xs):
    def prod(xs):
        p = array_reduce(lambda x, y: x * y, xs, ())
        return array_to_scalar(p)
    jy, bprop = J(prod)(J(xs))
    return Jinv(jy), bprop(1.0)


@infer_std(
    (f64, InferenceError)
)
def test_J_bprop_invalid(x):
    def f(x):
        return x * x
    _, bprop = J(f)(J(x))
    return bprop(1.0, 1.0)


@infer_std(
    (f64, (f64, f64))
)
def test_J_return_function(x):
    def f(y):
        return y * y

    def g():
        return f

    jg, _ = J(g)()
    jy, bprop = jg(J(x))
    _, dy = bprop(1.0)
    return Jinv(jy), dy


@infer_std(
    (f32, f32, f32),
    (i16, i16, i16),
)
def test_grad(x, y):
    def f(x, y):
        return x * (y + x)
    return grad(f)(x, y)


_f16 = to_abstract_test(f16)


@infer_std(
    (i64, i64),
    (f32, f32),
    (f64, f64),
)
def test_grad_cast(x):
    def f(x):
        return scalar_cast(x, _f16)

    return grad(f)(x)


@infer_std(
    (af16_of(2, 5), af16_of(2, 5), af16_of(2, 5)),
)
def test_grad_reduce(xs, ys):
    def f(xs, ys):
        return array_reduce(scalar_add, xs * ys, ())

    return grad(f)(xs, ys)
