
import pytest
import operator
import numpy as np

from types import SimpleNamespace

from myia.api import scalar_pipeline, standard_pipeline
from myia.composite import hyper_add, zeros_like
from myia.debug.traceback import print_inference_error
from myia.dtype import Array as A, Int, Float, TypeType, External, \
    Number, Class
from myia.hypermap import HyperMap
from myia.infer import \
    ANYTHING, InferenceError, register_inferrer
from myia.ir import MultitypeGraph
from myia.pipeline import pipeline_function
from myia.prim import Primitive, ops as P
from myia.prim.shape_inferrers import TupleShape, ListShape, ClassShape, \
    NOSHAPE
from myia.prim.py_implementations import \
    scalar_add, scalar_mul, scalar_lt, tail, list_map, hastype, \
    typeof, scalar_usub, dot, distribute, shape, array_map, \
    array_scan, array_reduce, reshape, partial as myia_partial, identity, \
    bool_and, bool_or, switch, scalar_to_array, broadcast_shape, \
    tuple_setitem, list_setitem, scalar_cast
from myia.utils import RestrictedVar

from .common import B, T, L, F, i16, i32, i64, u64, f16, f32, f64, \
    li32, li64, lf64, ai16, ai32, ai64, af16, af32, af64, Nil, \
    Point, Point_t, Point3D, Point3D_t, mysum


def t(tt):
    return {'type': tt}


def ai64_of(*shp):
    return {'type': ai64, 'shape': shp}


def ai32_of(*shp):
    return {'type': ai32, 'shape': shp}


def af64_of(*shp):
    return {'type': af64, 'shape': shp}


def af32_of(*shp):
    return {'type': af32, 'shape': shp}


def af16_of(*shp):
    return {'type': af16, 'shape': shp}


########################
# Temporary primitives #
########################


pyimpl_test = {}
value_inferrer_cons_test = {}
type_inferrer_cons_test = {}


def _test_op(cls):
    import inspect
    op = Primitive(cls.__name__)
    nargs = len(inspect.getargspec(cls.impl).args)
    pyimpl_test[op] = cls.impl
    for method in dir(cls):
        pfx = 'infer_'
        if method.startswith(pfx):
            track = method[len(pfx):]
            if track == 'type':
                cons = type_inferrer_cons_test
            elif track == 'value':
                cons = value_inferrer_cons_test
            else:
                raise Exception(f'Unknown track to infer: {track}')
            inffn = getattr(cls, method)
            register_inferrer(op, nargs=nargs, constructors=cons)(inffn)
    return op


# Ternary arithmetic op


@_test_op
class _tern:
    def impl(x, y, z):
        return x + y + z

    async def infer_type(track, x, y, z):
        return await track.will_check((Int, Float), x, y, z)


# Coercion


@_test_op
class _to_i64:
    def impl(x):
        return int(x)

    async def infer_type(track, x):
        return Int[64]


# Unification tricks


@_test_op
class _unif1:
    def impl(x):
        return x

    async def infer_type(track, x):
        rv = RestrictedVar({i16, f32})
        return track.engine.loop.create_var(rv, None, 0)


@_test_op
class _unif2:
    def impl(x):
        return x

    async def infer_type(track, x):
        rv = RestrictedVar({i16, f64})
        return track.engine.loop.create_var(rv, None, 0)


infer_pipeline = scalar_pipeline.select(
    'parse', 'infer'
).configure({
    'py_implementations': pyimpl_test,
    'infer.tracks.value.max_depth': 10,
    'infer.tracks.value.constructors': value_inferrer_cons_test,
    'infer.tracks.type.constructors': type_inferrer_cons_test,
})


infer_pipeline_std = standard_pipeline.select(
    'parse', 'infer'
).configure({
    'py_implementations': pyimpl_test,
    'infer.tracks.value.max_depth': 10,
    'infer.tracks.value.constructors': value_inferrer_cons_test,
    'infer.tracks.type.constructors': type_inferrer_cons_test,
})


def parse_test_spec(tests_spec):

    tests = []

    for main_track, ts in tests_spec.items():
        if not isinstance(ts, list):
            ts = [ts]
        for t in ts:
            test = []
            for entry in t:
                if isinstance(entry, dict) or entry is InferenceError:
                    test.append(entry)
                else:
                    test.append({main_track: entry})
            tests.append((main_track, test))

    return tests


def inferrer_decorator(pipeline):
    def infer(**tests_spec):

        tests = parse_test_spec(tests_spec)

        def decorate(fn):
            def run_test(spec):
                main_track, (*args, expected_out) = spec

                print('Args:')
                print(args)

                required_tracks = [main_track]

                def out():
                    pip = pipeline.configure({
                        'infer.required_tracks': required_tracks
                    })

                    res = pip.make()(input=fn, argspec=args)
                    rval = res['inference_results']

                    print('Output of inferrer:')
                    print(rval)
                    return rval

                print('Expected:')
                print(expected_out)

                if isinstance(expected_out, type) \
                        and issubclass(expected_out, Exception):
                    try:
                        out()
                    except InferenceError as e:
                        print_inference_error(e)
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


@infer(type=[(i64, i64)], value=[(89, 89)])
def test_identity(x):
    return x


@infer(type=[(i64,)], value=[(16,)])
def test_constants_int():
    return 2 * 8


@infer(type=[(f64,)], value=[(12.0,)])
def test_constants_float():
    return 1.5 * 8.0


@infer(type=[(f64,)], value=[(12.0,)])
def test_constants_intxfloat():
    return 8 * 1.5


@infer(type=[(f64,)], value=[(12.0,)])
def test_constants_floatxint():
    return 1.5 * 8


@infer(type=type_signature_arith_bin)
def test_prim_mul(x, y):
    return x * y


@infer(type=[
    (i64, i64, i64, i64),
    (f64, f64, f64, f64),
    # Three different inconsistent patterns below
    (f64, f64, i64, InferenceError),
    (i64, f64, f64, InferenceError),
    (f64, f64, i64, InferenceError),
    # Test too few/too many arguments below
    (i64, InferenceError),
    (i64, i64, i64, i64, InferenceError),
])
def test_prim_tern(x, y, z):
    return _tern(x, y, z)


@infer(type=[(i64, i64), (f64, f64), (B, InferenceError)])
def test_prim_usub(x):
    return -x


@infer_std(type=[
    (i64, InferenceError),
    (f32, f32),
    (f64, f64),
    (af64_of(2, 5), af64),
    (B, InferenceError)
])
def test_prim_log(x):
    return np.log(x)


@infer(
    type=[
        (B, f64, f64, f64),
        (B, f64, i64, InferenceError),
        ({'value': True}, f64, i64, f64),
        ({'value': False}, f64, i64, i64),
        # Note: scalar_pipeline will not convert i64 to bool,
        # so the following is an InferenceError even though it
        # will work with the standard_pipeline
        (i64, f64, f64, InferenceError),
    ],
    value=[
        (True, 7, 4, 49),
        (False, 7, 4, 16),
        ({'type': B, 'value': ANYTHING}, 7, 4, ANYTHING),
    ]
)
def test_if(c, x, y):
    if c:
        return x * x
    else:
        return y * y


@infer(type=type_signature_arith_bin)
def test_if2(x, y):
    if x > y:
        return x
    else:
        return y


@infer(
    type=[
        (i64, i64, i64),
        (i64, f64, f64),
        (f64, f64, f64),
        ({'value': 1_000_000}, i64, i64)
    ],
    value=[
        (2, 3, 27),
        (1_000_000, 3, ANYTHING)
    ]
)
def test_while(x, y):
    rval = y
    while x > 0:
        rval = rval * y
        x = x - 1
    return rval


@infer(
    type=[
        (li64, i64, i64),
        (li64, f64, InferenceError),
        (i64, i64, InferenceError),
        (T[i64, i64, i64], i64, i64),
        (T[i64, f64, i64], i64, InferenceError),
    ]
)
def test_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


@infer(type=(i64, f64, T[i64, f64]))
def test_nullary_closure(x, y):
    def make(z):
        def inner():
            return z
        return inner
    a = make(x)
    b = make(y)
    return a(), b()


@infer(type=(i64, f64, T[i64, f64]))
def test_merge_point(x, y):
    def mul2():
        return scalar_mul
    m = mul2()
    return m(x, x), m(y, y)


@infer(type=[(i64, InferenceError)])
def test_not_enough_args_prim(x):
    return scalar_mul(x)


@infer(type=[(i64, i64, i64, InferenceError)])
def test_too_many_args_prim(x, y, z):
    return scalar_mul(x, y, z)


@infer(type=[(i64, InferenceError)])
def test_not_enough_args(x):
    def g(x, y):
        return x * y
    return g(x)


@infer(type=[(i64, i64, InferenceError)])
def test_too_many_args(x, y):
    def g(x):
        return x * x
    return g(x, y)


@infer(type=(i64, f64, T[i64, f64]),
       shape=[(t(i64), t(f64), TupleShape((NOSHAPE, NOSHAPE))),
              (t(T[i64, i64]), t(f64),
               TupleShape((TupleShape((NOSHAPE, NOSHAPE)), NOSHAPE)))])
def test_tup(x, y):
    return (x, y)


@infer(
    type=[
        (T[i64, f64], i64),
        (lf64, InferenceError),
        (af64_of(2, 5), InferenceError),
        (i64, InferenceError),
    ],
    value=[
        ((), 0),
        ((1,), 1),
        ((1, 2), 2),
    ]
)
def test_tuple_len(xs):
    return P.tuple_len(xs)


@infer(
    type=[
        (T[i64, f64], InferenceError),
        (lf64, i64),
        (af64_of(2, 5), InferenceError),
        (i64, InferenceError),
    ],
)
def test_list_len(xs):
    return P.list_len(xs)


@infer(
    type=[
        (T[i64, f64], InferenceError),
        (lf64, InferenceError),
        (af64_of(2, 5), i64),
        (i64, InferenceError),
    ],
)
def test_array_len(xs):
    return P.array_len(xs)


@infer(type=[(T[i64, f64], T[f64]),
             (T[f64, i64], T[i64]),
             (T[()], InferenceError),
             (f64, InferenceError)],
       shape=[(t(T[f64, i64]), TupleShape((NOSHAPE,))),
              (t(T[i64, T[i64, f64]]),
               TupleShape((TupleShape((NOSHAPE, NOSHAPE)),)))])
def test_tail_tuple(tup):
    return tail(tup)


@infer(type=[(i64, f64, i64), (f64, i64, f64)],
       shape=(t(i64), t(f64), NOSHAPE))
def test_tuple_getitem(x, y):
    return (x, y)[0]


@infer(
    type=[
        (li64, i64, i64),
        (lf64, i64, f64),
        (lf64, f64, InferenceError),
        (f64, i64, InferenceError),
        (T[i64, f64], i64, InferenceError)
    ]
)
def test_list_getitem(xs, i):
    return xs[i]


@infer(
    type=[
        (T[i64, i64], {'value': 1}, f64, T[i64, f64]),
        (T[i64, i64, f64], {'value': 1}, f64, T[i64, f64, f64]),
        (T[i64], {'value': 1}, f64, InferenceError),
        (T[i64], {'type': f64, 'value': 0}, f64, InferenceError),
        (T[i64], i64, f64, InferenceError),
    ]
)
def test_tuple_setitem(xs, idx, x):
    return tuple_setitem(xs, idx, x)


@infer(
    type=[
        (li64, i64, i64, li64),
        (li64, f64, i64, InferenceError),
        (li64, i64, f64, InferenceError),
    ]
)
def test_list_setitem(xs, idx, x):
    return list_setitem(xs, idx, x)


@infer(type=(i64, f64, T[i64, f64]))
def test_multitype_function(x, y):
    def mul(a, b):
        return a * b
    return (mul(x, x), mul(y, y))


@infer(type=type_signature_arith_bin)
def test_closure(x, y):
    def mul(a):
        return a * x
    return mul(x) + mul(y)


@infer(
    type=[
        (i64, i64, i64, i64, T[i64, i64]),
        (f64, f64, f64, f64, T[f64, f64]),
        (i64, i64, f64, f64, T[i64, f64]),
        (i64, f64, f64, f64, InferenceError),
        (i64, i64, i64, f64, InferenceError),
    ]
)
def test_return_closure(w, x, y, z):
    def mul(a):
        def clos(b):
            return a * b
        return clos
    return (mul(w)(x), mul(y)(z))


@infer(type=[(i64, i64), (f64, f64)])
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


@infer(type=[(i64, B), (f64, B)])
def test_even_odd(n):
    return even(n)


@infer(type=[(i64, i64), (f64, f64)])
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
    type=[
        (i64, i64, i64, i64),
        (i64, f64, f64, f64)
    ]
)
def test_choose_prim(i, x, y):

    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_mul

    return choose(i)(x, y)


@infer(
    type=[
        (i64, i64, i64, InferenceError),
        ({'value': 0}, i64, i64, i64),
        ({'value': 1}, i64, i64, B),
    ]
)
def test_choose_prim_incompatible(i, x, y):

    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_lt

    return choose(i)(x, y)


@infer(
    type=[
        (i64, i64, i64, InferenceError),
        ({'value': 0}, i64, i64, i64),
        ({'value': 1}, i64, i64, B),
    ]
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
    type=[
        (i64, i64, i64),
        (i64, f64, f64)
    ],
    shape=[
        (t(i64), t(i64), NOSHAPE)
    ]
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


@infer(
    type=[
        (i64, i64)
    ]
)
def test_hof(x):

    def double(x):
        return x + x

    def square(x):
        return x * x

    def hof(f, tup):
        return f(tup[0]) + f(tup[1])

    return hof(double, (x + 1, x + 2)) + hof(square, (x + 3, x + 4))


@infer(
    type=[
        (i64, i64, i64),
        (i64, f64, InferenceError)
    ],
    value=[
        (-1, 3, 36),
        (1, 3, 6),
        ({'type': i64, 'value': ANYTHING}, 3, ANYTHING)
    ]
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


@infer(
    type=[
        (i64, T[T[i64, i64], T[B, B]])
    ]
)
def test_hof_3(x):

    def double(x):
        return x + x

    def is_zero(x):
        return x == 0

    def hof(f, tup):
        return (f(tup[0]), f(tup[1]))

    return (hof(double, (x + 1, x + 2)), hof(is_zero, (x + 3, x + 4)))


@infer(
    type=[
        (i64, i64, InferenceError),
        ({'value': -1}, i64, i64),
        ({'value': 1}, i64, T[i64, i64]),
    ]
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
    type=[
        (B, B, i64, i64, i64),
        (B, B, f64, f64, InferenceError),
        ({'value': True}, B, Nil, i64, i64),
        (B, {'value': True}, f64, f64, f64),
        (B, {'value': True}, i64, f64, InferenceError),
    ]
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


@infer(
    type=[
        (i64, i64, i64)
    ]
)
def test_func_arg(x, y):
    def g(func, x, y):
        return func(x, y)

    def h(x, y):
        return x + y
    return g(h, x, y)


@infer(
    type=[
        (i64, InferenceError)
    ]
)
def test_func_arg3(x):
    def g(func, x):
        z = func + x
        return func(z)

    def h(x):
        return x

    return g(h, x)


@infer(
    type=[
        (i64, i64),
        (f64, f64),
    ]
)
def test_func_arg4(x):
    def h(x):
        return x

    def g(fn, x):
        return fn(h, x)

    def t(fn, x):
        return fn(x)

    return g(t, x)


@infer(type=(i64,), value=(4,))
def test_closure_deep():
    def g(x):
        def h():
            return x * x
        return h
    return g(2)()


@infer(
    type=[
        (i64, i64, i64)
    ],
    value=[
        (5, 7, 15)
    ]
)
def test_closure_passing(x, y):
    def adder(x):
        def f(y):
            return x + y
        return f

    a1 = adder(1)
    a2 = adder(2)

    return a1(x) + a2(y)


@infer(type=[(B, B), (i64, InferenceError)])
def test_not(x):
    return not x


@infer(value=[(2, 2, 8), (2, 3, 13)])
def test_cover_limitedvalue_eq(x, y):

    def square(x):
        return x * x

    return square(x) + square(y)


@infer(
    type=[
        (li64, lf64, T[li64, lf64]),
        (li64, f64, InferenceError),
    ]
)
def test_list_map(xs, ys):

    def square(x):
        return x * x

    return list_map(square, xs), list_map(square, ys)


@infer(
    type=[
        (li64, li64, li64),
        (li64, lf64, InferenceError),
    ]
)
def test_list_map2(xs, ys):

    def mulm(x, y):
        return x * -y

    return list_map(mulm, xs, ys)


@infer(
    type=[(i64, B)],
    value=[(t(i64), True),
           (t(f64), False)]
)
def test_hastype_simple(x):
    return hastype(x, i64)


@infer(
    type=[
        (i64, i64, InferenceError),
        (i64, {'type': TypeType, 'value': Int[64]}, B),
    ],
    value=[
        ({'type': i64, 'value': ANYTHING},
         {'type': TypeType, 'value': ANYTHING}, InferenceError),
        ({'type': i64, 'value': ANYTHING},
         {'type': TypeType, 'value': i64}, {'value': True}),
        ({'type': f64, 'value': ANYTHING},
         {'type': TypeType, 'value': i64}, {'value': False}),
        ({'type': T[i64, i64], 'value': ANYTHING},
         {'type': TypeType, 'value': T[i64, i64]}, {'value': True}),
        ({'type': T[i64, i64], 'value': ANYTHING},
         {'type': TypeType, 'value': T[Number, Number]}, {'value': True}),
    ]
)
def test_hastype(x, y):
    return hastype(x, y)


@infer(
    type=[(i64, TypeType)],
    value=[(t(i64), i64),
           (t(f64), f64)]
)
def test_typeof(x):
    return typeof(x)


@infer(
    type=[
        (i64, i64),
        (f64, i64),
        (T[i64, i64], i64),
        (T[i64, T[f64, i64]], i64),
        (li64, f64),
        (T[i64, li64], i64),
        (Point_t, i64),
        (Point3D_t, i64),
    ],
    value=[
        (5, 5),
        (6.0, 6),
        ((5, 7, (3.2, 1.8)), 16),
        (Point(5, 7), 35),
        (Point3D(5, 7, 9), 0),
    ]
)
def test_hastype_2(x):

    def f(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return f(_to_i64(x))
        elif hastype(x, Point_t):
            return f(x.x) * f(x.y)
        elif hastype(x, Nil):
            return 0
        elif hastype(x, T):
            return f(x[0]) + f(tail(x))
        elif hastype(x, L):
            return 1.0
        else:
            return 0

    return f(x)


@infer_std(
    type=[
        (li64, i64, li64),
        (lf64, i64, InferenceError),
        ({'type': L[ai64], 'shape': ListShape((2, 3))}, i64, L[ai64]),
    ],
    shape=[(t(li64), t(i64), ListShape(NOSHAPE)),
           ({'type': L[ai64], 'shape': ListShape((2, 3))}, ai64_of(2, 3),
            ListShape((2, 3)))]
)
def test_map_2(xs, z):

    def adder(x):
        def f(y):
            return x + y
        return f

    return list_map(adder(z), xs)


def _square(x):
    return x * x


@infer(type=(InferenceError,))
def test_nonexistent_variable():
    return xxxx + yz  # noqa


class helpers:
    add = operator.add
    mul = operator.mul
    square = _square


class data:
    a25 = np.ones((2, 5))


@infer(
    type=[
        (i64, i64, T[i64, i64]),
        (i64, f64, InferenceError),
    ],
    value=[
        (2, 3, (5, 36))
    ]
)
def test_getattr(x, y):
    a = helpers.add(x, y)
    b = helpers.mul(x, y)
    c = helpers.square(b)
    return a, c


@infer(
    type=[
        (i64, i64, T[i64, i64]),
        (i64, f64, T[i64, f64]),
    ]
)
def test_getattr_multitype(x, y):
    a = helpers.add(x, x)
    b = helpers.add(y, y)
    return a, b


@infer(
    shape=[
        ((2, 5),),
    ]
)
def test_getattr_shape():
    return data.a25


_getattr = getattr


@infer(
    type=[
        ({'value': 'add'}, i64, i64),
        ({'value': 'bad'}, i64, InferenceError),
        ({'value': 1234}, i64, InferenceError),
        (External[str], i64, InferenceError),
    ]
)
def test_getattr_flex(name, x):
    return _getattr(helpers, name)(x, x)


@infer(type=[
    (External[SimpleNamespace],
     {'type': External[str], 'value': 'surprise'},
     InferenceError)
])
def test_unknown_data(data, field):
    return _getattr(data, field)


@infer(type=[(i64, i64, i64), (f64, f64, f64)])
def test_method(x, y):
    return x.__add__(y)


@infer(type=[(i64, i64, InferenceError)])
def test_unknown_method(x, y):
    return x.unknown(y)


@infer(type=[(i64, InferenceError)])
def test_infinite_recursion(x):
    def ouroboros(x):
        return ouroboros(x - 1)

    return ouroboros(x)


@infer(type=[(i64, InferenceError)])
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


@infer(type=[(i64, InferenceError)])
def test_infinite_mutual_recursion(x):
    return ping()


@infer(type=[({'shape': (2, 3), 'type': ai16}, T[u64, u64])],
       value=[({'shape': (2, 3), 'type': ai16}, (2, 3)),
              ({'shape': (2, ANYTHING), 'type': ai16}, ANYTHING)])
def test_shape(ary):
    return shape(ary)


@infer(shape=[
    ({'value': Point(2, 3)}, ClassShape({'x': NOSHAPE, 'y': NOSHAPE})),
    ({'value': [np.ones((2, 3)), np.ones((2, 3))]}, ListShape((2, 3))),
    ({'value': [np.ones((2, 2)), np.ones((2, 3))]}, ListShape((2, ANYTHING))),
])
def test_shape2(val):
    return val


@infer(shape=[(af64_of(2, 3),
               af64_of(3, 4), (2, 4)),
              (af64_of(2),
               af64_of(3, 4), InferenceError),
              (af64_of(2, 2),
               af64_of(3, 4), InferenceError)],
       type=[(af64_of(2, 3),
              af64_of(3, 4), af64)])
def test_dot(a, b):
    return dot(a, b)


@infer(shape=[(ai32_of(4), {'type': T[u64, u64], 'value': (2, 4)}, (2, 4)),
              (ai32_of(4),
               {'type': T[u64, u64]},
               (ANYTHING, ANYTHING)),
              (ai32_of(4),
               {'type': T[u64, u64], 'value': (5, 2)},
               InferenceError),
              ({'type': ai32, 'shape': (4, 2)},
               {'type': T[u64], 'value': (4,)},
               InferenceError)],
       type=[
           (i32, {'value': (4,), 'type': T[u64]}, InferenceError),
           (ai32_of(1), {'value': (4,), 'type': T[u64]}, ai32),
           (li32, {'value': (4,), 'type': T[u64]}, InferenceError),
           (i32, {'value': (4,)}, InferenceError)
       ])
def test_distribute(v, shp):
    return distribute(v, shp)


@infer(shape=[(af16_of(1, 2, 3), {'type': T[u64], 'value': (6,)}, (6,)),
              (af16_of(1, 2, 3), {'type': T[u64]},
               (ANYTHING,)),
              (af16_of(2, 3), {'type': T[u64], 'value': (7,)},
               InferenceError)],
       type=[(af16_of(2, 3), T[u64], af16),
             (af16_of(2, 3), T[i64],
              InferenceError)])
def test_reshape(v, shp):
    return reshape(v, shp)


@infer(shape=[({'type': af32, 'shape': (3, 4)}, (3, 4))],
       type=[({'type': ai64, 'shape': (3, 4)}, ai64),
             ({'type': i64}, InferenceError)])
def test_array_map(ary):
    def f(v):
        return v + 1
    return array_map(f, ary)


@infer(shape=[(af32_of(3, 4), af32_of(3, 4), (3, 4)),
              (af32_of(3, 4), af32_of(3, 7), InferenceError),
              (af32_of(3, ANYTHING), af32_of(ANYTHING, 7), (3, 7)),
              (af32_of(3, ANYTHING), af32_of(ANYTHING, ANYTHING),
               (3, ANYTHING)),
              (af32_of(3, 4, 5), af32_of(3, 4), InferenceError)],
       type=[(ai64_of(7, 9), ai64_of(7, 9), ai64),
             (ai64_of(7, 9), i64, InferenceError),
             (i64, ai64_of(7, 9), InferenceError)])
def test_array_map2(ary1, ary2):
    def f(v1, v2):
        return v1 + v2
    return array_map(f, ary1, ary2)


@infer(shape=[(af32_of(3, 4), af32_of(3, 4), af32_of(3, 4), (3, 4)),
              (af32_of(3, 4), af32_of(3, 4), af32_of(3, 7), InferenceError),
              (af32_of(3, ANYTHING, 5, 6),
               af32_of(3, 4, 5, ANYTHING),
               af32_of(ANYTHING, ANYTHING, ANYTHING, 6),
               (3, 4, 5, 6)),
              (af32_of(3, ANYTHING, 5, 6),
               af32_of(3, 4, 5, ANYTHING),
               af32_of(ANYTHING, ANYTHING, ANYTHING, 7),
               InferenceError),
              (af32_of(3, 4, 5), af32_of(3, 4), af32_of(3, 4),
               InferenceError)],
       type=[(ai64_of(7, 9), ai64_of(7, 9), ai64_of(7, 9), ai64),
             (ai64_of(7, 9), ai64_of(7, 9), i64, InferenceError),
             (i64, ai64_of(7, 9), ai64_of(7, 9), InferenceError)])
def test_array_map3(ary1, ary2, ary3):
    def f(v1, v2, v3):
        return v1 + v2 + v3
    return array_map(f, ary1, ary2, ary3)


@infer(shape=[(ai64_of(3, 4), {'value': 1, 'type': u64}, (3, 4))],
       type=[
           (ai64_of(3, 4), {'value': 1, 'type': u64}, ai64),
           ({'type': i64}, {'value': 1, 'type': u64}, InferenceError),
           (af32_of(3, 4), {'value': 1, 'type': u64},
            InferenceError),
           (ai64_of(3, 4), {'value': 1}, InferenceError)
       ])
def test_array_scan(ary, ax):
    def f(a, b):
        return a + b
    return array_scan(f, 0, ary, ax)


@infer(
    type=[
        (ai64_of(7, 9), T[u64, u64], ai64),
        (ai64_of(7, 9), i64, InferenceError),
        (i64, T[u64, u64], InferenceError),
    ],
    shape=[
        (ai64_of(3, 4),
         {'type': T[u64, u64], 'value': (3, 1)},
         (3, 1)),

        (ai64_of(3, 4),
         {'type': T[u64, u64], 'value': (3, ANYTHING)},
         (3, ANYTHING)),

        (ai64_of(3, 4),
         {'type': T[u64, u64, u64], 'value': (3, 1, 1)},
         InferenceError),

        (ai64_of(3, 4),
         {'type': T[u64, u64], 'value': (4, 1)},
         InferenceError),

        (ai64_of(3, 4),
         {'type': T[u64], 'value': (4,)},
         (4,)),

        (ai64_of(3, 4),
         {'value': ()},
         ()),
    ]
)
def test_array_reduce(ary, shp):
    def f(a, b):
        return a + b
    return array_reduce(f, ary, shp)


@infer(type=[(i64, i64)],
       value=[(40, 42)],
       shape=[({'type': i64}, NOSHAPE)])
def test_partial_1(x):
    def f(a, b):
        return a + b
    f2 = myia_partial(f, 2)
    return f2(x)


@infer(type=[(i64, i64)])
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


@infer(type=[(i64, i64)],
       shape=[(ai64_of(6, 13), (6, 13))])
def test_identity_function(x):
    return identity(x)


@infer(type=[(B, B, B),
             (i64, B, InferenceError),
             (B, i64, InferenceError)])
def test_bool_and(x, y):
    return bool_and(x, y)


@infer(type=[(B, B, B),
             (i64, B, InferenceError),
             (B, i64, InferenceError)])
def test_bool_or(x, y):
    return bool_or(x, y)


@infer(
    type=[
        (B, i64, i64, i64),
        (i64, i64, i64, InferenceError),
        (B, i64, f64, InferenceError),
        ({'value': True}, i64, f64, i64),
        ({'value': False}, i64, f64, f64),
    ],
    value=[
        (True, 1, 2, 1),
        (False, 1, 2, 2),
        ({'type': B, 'value': ANYTHING}, 1, 2, ANYTHING),
    ],
    shape=[
        ({'type': B},
         ai64_of(6, 13),
         ai64_of(6, 13),
         (6, 13)),

        ({'type': B},
         ai64_of(6, 13),
         ai64_of(6, 14),
         InferenceError),

        ({'type': B, 'value': True},
         ai64_of(6, 13),
         ai64_of(6, 14),
         (6, 13)),

        ({'type': B, 'value': False},
         ai64_of(6, 13),
         ai64_of(6, 14),
         (6, 14)),
    ]
)
def test_switch(c, x, y):
    return switch(c, x, y)


@infer(
    type=[
        (i64, {'value': i64}, i64),
        (i64, {'value': i16}, i16),
        (f64, {'value': i16}, i16),
        (f16, {'value': f32}, f32),
        (f16, TypeType, InferenceError),
        (f16, {'value': B}, InferenceError),
        (B, {'value': f32}, InferenceError),
    ]
)
def test_scalar_cast(x, t):
    return scalar_cast(x, t)


@infer(type=[(i64, ai64),
             (f64, af64),
             (af64_of(9, 7), InferenceError),
             (T[i64], InferenceError)],
       shape=[({'type': i64}, ())])
def test_scalar_to_array(x):
    return scalar_to_array(x)


@infer(type=[
    (T[u64], T[u64], T[u64]),
    (T[u64, u64], T[u64], T[u64, u64]),
    (T[u64], T[u64, u64], T[u64, u64]),
    (T[i64], T[u64], InferenceError),
    (T[u64], T[i64], InferenceError),
    (i64, i64, InferenceError),
])
def test_broadcast_shape(xs, ys):
    return broadcast_shape(xs, ys)


@infer(type=[
    (i64, i64, InferenceError),
    (F[[i64], f64], i64, f64),
    (F[[f64], f64], i64, InferenceError),
])
def test_call_argument(f, x):
    return f(x)


@infer(type=[
    (F[[F[[f64], f64]], f64], f64),
])
def test_call_argument_higher_order(f):
    def g(y):
        return y + y
    return f(g)


@infer(type=[
    (i64, i16),
])
def test_unif_tricks(x):
    # unif1 is i16 or f32
    # unif2 is i16 or f64
    # a + b requires same type for a and b
    # Therefore unif1 and unif2 are i16
    a = _unif1(x)
    b = _unif2(x)
    return a + b


@infer(type=[
    (i64, InferenceError),
])
def test_unif_tricks_2(x):
    # unif1 is i16 or f32
    # Both are possible, so we raise an error due to ambiguity
    a = _unif1(x)
    return a + a


@infer(
    value=[
        (2, 3, 4, 90)
    ],
    type=[
        (i64, i64, i64, i64),
        (f64, f64, f64, InferenceError),
    ]
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
    type=[
        (ai64_of(7, 9), ai64_of(9, 2), ai64),
        (af64_of(7, 9), af64_of(7, 9), af64),
        (f64, f64, InferenceError),
    ],
    shape=[
        (ai64_of(2, 5), ai64_of(5, 3), (2, 3)),
        (af64_of(2, 5), af64_of(5, 3), InferenceError),
        (ai64_of(2, 5), ai64_of(2, 5), InferenceError),
        (af64_of(2, 5), af64_of(2, 5), (2, 5)),
    ]
)
def test_multitype_2(x, y):
    return mystery(x, y)


def test_forced_type():

    @pipeline_function
    def mod(self, graph):
        # Force the inferred tyoe of the output to be f64
        graph.output.inferred['type'] = f64
        return graph

    def fn(x, y):
        return x + y

    pip = infer_pipeline.insert_before('infer', mod=mod)

    for argspec in [[{'type': i64}, {'type': i64}],
                    [{'type': i64}, {'type': f64}]]:

        results = pip.run(input=fn, argspec=argspec)
        rval = results['inference_results']

        assert rval['type'] == f64


def test_forced_function_type():

    @pipeline_function
    def mod(self, graph):
        # Force the inferred tyoe of scalar_add to be (i64,i64)->f64
        scalar_add = graph.output.inputs[0]
        scalar_add.inferred['type'] = F[[i64, i64], f64]
        return graph

    def fn(x, y):
        return x + y

    pip = infer_pipeline.insert_before('infer', mod=mod)

    # Test correct

    results = pip.run(
        input=fn,
        argspec=[{'type': i64}, {'type': i64}]
    )
    rval = results['inference_results']

    assert rval['type'] == f64

    # Test mismatch

    with pytest.raises(InferenceError):
        results = pip.run(
            input=fn,
            argspec=[{'type': i64}, {'type': f64}]
        )

    # Test narg mismatch

    def fn2(x):
        return fn(x)

    with pytest.raises(InferenceError):
        results = pip.run(
            input=fn2,
            argspec=[{'type': i64}]
        )


###########################
# Using standard_pipeline #
###########################


@infer_std(
    type=[
        (i64, i64, i64),
        (ai64_of(7, 9), ai64_of(7, 9), ai64),
        (ai64_of(7, 9), i64, ai64),
        (i64, ai64_of(7, 9), ai64),
        (i64, f64, InferenceError),
        ({'type': i64, 'value': 3}, ai64_of(7, 9), ai64)
    ],
    shape=[
        (ai64_of(2, 5), ai64_of(2, 5), (2, 5)),
        (ai64_of(2, 5), ai64_of(2, 1), (2, 5)),
        (ai64_of(1, 5), ai64_of(2, 1), (2, 5)),
        (ai64_of(5,), ai64_of(2, 1), (2, 5)),
        (ai64_of(2, 3, 4), ai64_of(3, 4), (2, 3, 4)),
        (ai64_of(5,), ai64_of(2,), InferenceError),
        ({'type': i64}, ai64_of(2, 5), (2, 5)),
        (ai64_of(2, 5), {'type': i64}, (2, 5)),
    ]
)
def test_add_std(x, y):
    return x + y


@infer_std(type=[(i64, i64, i64),
                 (ai64_of(7, 9), i64, InferenceError)])
def test_max_std(x, y):
    if x > y:
        return x
    else:
        return y


@infer_std(
    type=[
        (f64, f64),
        (i64, i64),
        (af32_of(2, 5), af32),
    ]
)
def test_add1_stdx(x):
    return 1 + x


def _add(x, y):
    return x + y


@infer_std(
    type=[
        (f64, f64),
        (i64, i64),
    ]
)
def test_add1_std_indirect(x):
    return _add(1, x)


def _interference_helper(x):
    if hastype(x, T):
        return x[0]
    else:
        return x


@infer(
    type=[
        (i64, i64),
        (f64, f64),
    ]
)
def test_add1_hastype_interference(x):
    return x + _interference_helper(1)


@infer(
    type=[
        (f16, f32, f64, f32),
    ]
)
def test_hastype_interference(x, y, z):
    if hastype(1, i32):
        return x
    elif hastype(1, i64):
        return y
    else:
        return z


@infer(
    type=[
        (Point_t, i64),
    ],
    value=[
        (Point(3, 4), 7),
        ({'type': Point_t, 'value': ANYTHING}, ANYTHING),
    ],
    shape=[
        (t(Point_t), NOSHAPE)
    ]
)
def test_class(pt):
    return pt.x + pt.y


@infer(
    type=[
        (Point_t, i64),
    ],
    value=[
        (Point(3, 4), 5),
    ]
)
def test_dataclass_method(pt):
    return pt.abs()


@infer(
    type=[
        (i64, i64, i64, i64, Point_t),
        (f64, f64, f64, f64, InferenceError),
    ],
    value=[
        (1, 2, 3, 4, Point(4, 6)),
    ],
    shape=[
        (t(i64), t(i64), t(i64), t(i64),
         ClassShape({'x': NOSHAPE, 'y': NOSHAPE}))
    ]
)
def test_dataclass_inst(x1, y1, x2, y2):
    pt1 = Point(x1, y1)
    pt2 = Point(x2, y2)
    return Point(pt1.x + pt2.x, pt1.y + pt2.y)


@infer(type=[(Point_t, InferenceError)])
def test_dataclass_wrong_field(pt):
    return pt.z


hyper_map = HyperMap()
hyper_map_notuple = HyperMap(nonleaf=(A, Class))
hyper_map_nobroadcast = HyperMap(broadcast=False)


@infer(
    type=[
        (i64, i64, i64),
        (f64, f64, f64),
        (lf64, lf64, lf64),
        (L[lf64], L[lf64], L[lf64]),
        (lf64, f64, InferenceError),
        (L[f64], L[lf64], InferenceError),
        (T[i64, f64], T[i64, f64], T[i64, f64]),
        (Point_t, Point_t, Point_t),
        (ai64_of(2, 5), ai64_of(2, 5), ai64),
        (i64, f64, InferenceError),
        (lf64, f64, InferenceError),
        (ai64_of(2, 5), af64_of(2, 5), InferenceError),
    ],
    value=[
        (1, 2, 3),
        (4.5, 7.5, 12.0),
        (Point(1, 2), Point(3, 4), Point(4, 6)),
    ],
    shape=[
        (ai64_of(2, 5), ai64_of(2, 5), (2, 5)),
        (ai64_of(2, 5), ai64_of(2, 1), (2, 5)),
        (ai64_of(2, 5), {'type': i64}, (2, 5)),
    ]
)
def test_hyper_map(x, y):
    return hyper_map(scalar_add, x, y)


@infer(
    type=[
        (T[i64, f64], T[i64, f64], InferenceError),
    ],
)
def test_hyper_map_notuple(x, y):
    return hyper_map_notuple(scalar_add, x, y)


@infer(
    shape=[
        (ai64_of(2, 5), ai64_of(2, 5), (2, 5)),
        (ai64_of(2, 5), ai64_of(2, 1), InferenceError),
        (ai64_of(2, 5), {'type': i64}, InferenceError),
    ]
)
def test_hyper_map_nobroadcast(x, y):
    return hyper_map_nobroadcast(scalar_add, x, y)


@infer(
    type=[
        (i64, i64, i64),
        (f64, f64, f64),
        (lf64, lf64, lf64),
        (T[i64, f64], T[i64, f64], T[i64, f64]),
        (Point_t, Point_t, Point_t),
        (ai64_of(2, 5), ai64_of(2, 5), ai64),
    ],
    value=[
        (1, 2, 3),
        (4.5, 7.5, 12.0),
        (Point(1, 2), Point(3, 4), Point(4, 6)),
    ],
    shape=[
        (ai64_of(2, 5), ai64_of(2, 5), (2, 5)),
        (ai64_of(2, 5), ai64_of(2, 1), (2, 5)),
        (ai64_of(2, 5), {'type': i64}, (2, 5)),
    ]
)
def test_hyper_add(x, y):
    return hyper_add(x, y)


@infer(
    type=[
        (i64, i64),
        (f64, f64),
        (lf64, lf64),
        (T[i64, f64], T[i64, f64]),
        (Point_t, Point_t),
        (ai64_of(2, 5), ai64),
        (af32_of(2, 5), af32),
    ],
    value=[
        (1, 0),
        ((2, 3.0), (0, 0.0)),
        (Point(1, 2), Point(0, 0)),
    ],
    shape=[
        (ai64_of(2, 5), (2, 5)),
        (af32_of(2, 5), (2, 5)),
    ]
)
def test_zeros_like(x):
    return zeros_like(x)
