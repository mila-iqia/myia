
import operator
import numpy as np

from functools import partial
from pytest import mark
from types import SimpleNamespace

from myia.api import standard_pipeline
from myia.infer import \
    ANYTHING, InferenceError, register_inferrer
from myia.dtype import Array as A, Bool, Int, Float, Tuple as T, List as L, \
    Type, UInt, External
from myia.prim import Primitive
from myia.prim.py_implementations import \
    scalar_add, scalar_mul, scalar_lt, head, tail, list_map, hastype, \
    typeof, scalar_usub, dot, distribute, shape, array_map, array_scan, \
    array_reduce, reshape, partial as myia_partial, identity


B = Bool()

i16 = Int(16)
i32 = Int(32)
i64 = Int(64)

u64 = UInt(64)

f16 = Float(16)
f32 = Float(32)
f64 = Float(64)

li16 = L(Int(16))
li32 = L(Int(32))
li64 = L(Int(64))

lf16 = L(Float(16))
lf32 = L(Float(32))
lf64 = L(Float(64))

ai16 = A(i16)
ai32 = A(i32)
ai64 = A(i64)

af16 = A(f16)
af32 = A(f32)
af64 = A(f64)

Nil = T()


########################
# Temporary primitives #
########################


pyimpl_test = {}
value_inferrer_cons_test = {}
type_inferrer_cons_test = {}

type_inferrer_test = partial(register_inferrer,
                             constructors=type_inferrer_cons_test)
value_inferrer_test = partial(register_inferrer,
                              constructors=value_inferrer_cons_test)


# Ternary arithmetic op

_tern = Primitive('tern')


def impl_tern(x, y, z):
    return x + y + z


pyimpl_test[_tern] = impl_tern


@type_inferrer_test(_tern, nargs=3)
async def infer_type_tern(track, x, y, z):
    ret_t = await track.assert_same(x, y, z)
    assert isinstance(ret_t, (Int, Float))
    return ret_t


# Coercion

_to_i64 = Primitive('to_i64')


def impl_to_i64(x):
    return int(x)


pyimpl_test[_to_i64] = impl_to_i64


@type_inferrer_test(_to_i64, nargs=1)
async def infer_type_to_i64(track, x):
    return Int(64)


infer_pipeline = standard_pipeline.select(
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


def infer(**tests_spec):

    tests = parse_test_spec(tests_spec)

    def decorate(fn):
        def run_test(spec):
            main_track, (*args, expected_out) = spec

            print('Args:')
            print(args)

            required_tracks = [main_track]

            def out():
                pip = infer_pipeline.configure({
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
                    pass
                else:
                    raise Exception(
                        f'Expected {expected_out}, got: (see stdout).'
                    )
            else:
                assert out() == expected_out

        m = mark.parametrize('spec', list(tests))(run_test)
        m.__orig__ = fn
        return m

    return decorate


type_signature_arith_bin = [
    (i64, i64, i64),
    (f64, f64, f64),
    (i64, f64, InferenceError),
    (B, B, InferenceError),
]


@infer(type=[(f64,)], value=[(12.0,)])
def test_constants():
    return 1.5 * 8.0


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


@infer(
    type=[
        (B, f64, f64, f64),
        (B, f64, i64, InferenceError),
        ({'value': True}, f64, i64, f64),
        ({'value': False}, f64, i64, i64),
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
        (f64, f64, InferenceError),
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
    ]
)
def test_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


@infer(type=(i64, f64, T(i64, f64)))
def test_nullary_closure(x, y):
    def make(z):
        def inner():
            return z
        return inner
    a = make(x)
    b = make(y)
    return a(), b()


@infer(type=(i64, f64, T(i64, f64)))
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


@infer(type=(i64, f64, T(i64, f64)))
def test_tup(x, y):
    return (x, y)


@infer(type=[(T(i64, f64), i64),
             (T(f64, i64), f64),
             (T(), InferenceError),
             (f64, InferenceError)])
def test_head_tuple(tup):
    return head(tup)


@infer(type=[(T(i64, f64), T(f64)),
             (T(f64, i64), T(i64)),
             (T(), InferenceError),
             (f64, InferenceError)])
def test_tail_tuple(tup):
    return tail(tup)


@infer(type=[(i64, f64, i64), (f64, i64, f64)])
def test_getitem_tuple(x, y):
    return (x, y)[0]


@infer(
    type=[
        (li64, i64, i64),
        (lf64, i64, f64),
        (lf64, f64, InferenceError),
        (f64, i64, InferenceError),
        (T(i64, f64), i64, InferenceError)
    ]
)
def test_getitem_list(xs, i):
    return xs[i]


@infer(type=(i64, f64, T(i64, f64)))
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
        (i64, i64, i64, i64, T(i64, i64)),
        (f64, f64, f64, f64, T(f64, f64)),
        (i64, i64, f64, f64, T(i64, f64)),
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


@infer(type=[(i64, i64), (f64, InferenceError)])
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


@infer(type=[(i64, B), (f64, InferenceError)])
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
        ({'type': i64, 'shape': ()}, {'type': i64, 'shape': ()}, ())
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
        (i64, T(T(i64, i64), T(B, B)))
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
        ({'value': 1}, i64, T(i64, i64)),
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
        (li64, lf64, T(li64, lf64)),
        (li64, f64, InferenceError),
    ]
)
def test_list_map(xs, ys):

    def square(x):
        return x * x

    return list_map(square, xs), list_map(square, ys)


@infer(
    type=[(i64, B)],
    value=[({'type': i64}, True),
           ({'type': f64}, False)]
)
def test_hastype(x):
    return hastype(x, i64)


@infer(
    type=(i64, i64, InferenceError),
    value=(i64, {'type': Type, 'value': ANYTHING}, InferenceError),
)
def test_bad_hastype(x, y):
    return hastype(x, y)


@infer(
    type=[(i64, Type)],
    value=[({'type': i64}, i64),
           ({'type': f64}, f64)]
)
def test_typeof(x):
    return typeof(x)


@infer(
    type=[
        (i64, i64),
        (f64, i64),
        (T(i64, i64), i64),
        (T(i64, T(f64, i64)), i64),
        (li64, f64),
        (T(i64, li64), InferenceError),
    ],
    value=[
        (5, 5),
        (6.0, 6),
        ((5, 7, (3.2, 1.8)), 16)
    ]
)
def test_hastype_2(x):
    i64, f64, T, Nil, L, _to_i64, hastype, head, tail

    def f(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return f(_to_i64(x))
        elif hastype(x, Nil):
            return 0
        elif hastype(x, T):
            return f(head(x)) + f(tail(x))
        elif hastype(x, L):
            return 1.0
        else:
            return 0

    return f(x)


@infer(
    type=[
        (li64, i64, li64),
        (lf64, i64, InferenceError),
    ]
)
def test_map_2(xs, z):

    def adder(x):
        def f(y):
            return x + y
        return f

    return list_map(adder(z), xs)


def _square(x):
    return x * x


helpers = SimpleNamespace(
    add=operator.add,
    mul=operator.mul,
    square=_square
)


data = SimpleNamespace(
    a25=np.ones((2, 5))
)


@infer(
    type=[
        (i64, i64, T(i64, i64)),
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
        (i64, i64, T(i64, i64)),
        (i64, f64, T(i64, f64)),
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
        (External(str), i64, InferenceError),
    ]
)
def test_getattr_flex(name, x):
    return _getattr(helpers, name)(x, x)


@infer(type=[
    (External(SimpleNamespace),
     {'type': External(str), 'value': 'surprise'},
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


@infer(type=[({'shape': (2, 3), 'type': ai16}, T(u64, u64))],
       value=[({'shape': (2, 3), 'type': ai16}, (2, 3)),
              ({'shape': (2, ANYTHING), 'type': ai16}, ANYTHING)])
def test_shape(ary):
    return shape(ary)


@infer(shape=[({'type': af64, 'shape': (2, 3)},
               {'type': af64, 'shape': (3, 4)}, (2, 4)),
              ({'type': af64, 'shape': (2,)},
               {'type': af64, 'shape': (3, 4)}, InferenceError),
              ({'type': af64, 'shape': (2, 2)},
               {'type': af64, 'shape': (3, 4)}, InferenceError)],
       type=[({'type': af64, 'shape': (2, 3)},
              {'type': af64, 'shape': (3, 4)}, af64)])
def test_dot(a, b):
    return dot(a, b)


@infer(shape=[({'type': i32}, {'value': (4, 2)}, (4, 2)),
              ({'type': i32}, {'type': T(u64, u64)}, (ANYTHING, ANYTHING)),
              ({'type': ai32, 'shape': (4,)}, {'value': (4, 2)}, (4, 2)),
              ({'type': ai32, 'shape': (4,)}, {'value': (5, 2)},
               InferenceError),
              ({'type': ai32, 'shape': (4, 2)}, {'value': (4,)},
               InferenceError)],
       type=[
           ({'type': i32}, {'value': (4,), 'type': T(u64)}, ai32),
           ({'type': ai32, 'shape': (1,)}, {'value': (4,), 'type': T(u64)},
            ai32),
           ({'type': li32}, {'value': (4,), 'type': T(u64)}, InferenceError),
           ({'type': i32}, {'value': (4,)}, InferenceError)
       ])
def test_distribute(v, shp):
    return distribute(v, shp)


@infer(shape=[({'type': af16, 'shape': (1, 2, 3)}, {'value': (6,)}, (6,)),
              ({'type': af16, 'shape': (1, 2, 3)}, {'type': T(u64)},
               (ANYTHING,)),
              ({'type': af16, 'shape': (2, 3)}, {'value': (7,)},
               InferenceError)],
       type=[({'type': af16, 'shape': (2, 3)}, {'type': T(u64)}, af16),
             ({'type': af16, 'shape': (2, 3)}, {'type': T(i64)},
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


@infer(shape=[({'type': ai64, 'shape': (3, 4)}, {'value': 1}, (3, 4))],
       type=[
           ({'type': ai64, 'shape': (3, 4)}, {'value': 1, 'type': u64}, ai64),
           ({'type': i64}, {'value': 1, 'type': u64}, InferenceError),
           ({'type': af32, 'shape': (3, 4)}, {'value': 1, 'type': u64},
            InferenceError),
           ({'type': ai64, 'shape': (3, 4)}, {'value': 1}, InferenceError)
       ])
def test_array_scan(ary, ax):
    def f(a, b):
        return a + b
    return array_scan(f, 0, ary, ax)


@infer(shape=[({'type': ai64, 'shape': (3, 4)}, {'value': 1}, (3,)),
              ({'type': ai64, 'shape': (3, 4)}, {'type': u64}, (ANYTHING,))])
def test_array_reduce(ary, ax):
    def f(a, b):
        return a + b
    return array_reduce(f, 0, ary, ax)


@infer(type=[(i64, i64)],
       value=[(40, 42)],
       shape=[({'type': i64, 'shape': ()}, ())])
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
       shape=[({'type': ai64, 'shape': (6, 13)}, (6, 13))])
def test_identity(x):
    return identity(x)
