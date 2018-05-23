
from functools import partial
from pytest import mark

from myia.api import parse
from myia.infer import \
    InferenceEngine, ANYTHING, InferenceError, register_inferrer

from myia.dtype import Bool, Int, Float, Tuple as T, List as L, Type
from myia.prim import Primitive
from myia.prim.py_implementations import \
    py_implementations as pyimpl, \
    add, mul, lt, head, tail, maplist, hastype, typeof, usub
from myia.prim.value_inferrers import \
    ValueTrack, value_inferrer_constructors
from myia.prim.type_inferrers import \
    TypeTrack, type_inferrer_constructors


B = Bool()

i16 = Int(16)
i32 = Int(32)
i64 = Int(64)

f16 = Float(16)
f32 = Float(32)
f64 = Float(64)

li16 = L(Int(16))
li32 = L(Int(32))
li64 = L(Int(64))

lf16 = L(Float(16))
lf32 = L(Float(32))
lf64 = L(Float(64))

Nil = T()


########################
# Temporary primitives #
########################


pyimpl_test = {**pyimpl}
value_inferrer_cons_test = {**value_inferrer_constructors}
type_inferrer_cons_test = {**type_inferrer_constructors}

type_inferrer_test = partial(register_inferrer,
                             constructors=type_inferrer_cons_test)
value_inferrer_test = partial(register_inferrer,
                              constructors=value_inferrer_cons_test)

value_track = partial(
    ValueTrack,
    implementations=pyimpl_test,
    constructors=value_inferrer_cons_test
)
type_track = partial(
    TypeTrack,
    constructors=type_inferrer_cons_test
)


# Ternary arithmetic op

_tern = Primitive('tern')


def impl_tern(x, y, z):
    return x + y + z


pyimpl_test[_tern] = impl_tern


@type_inferrer_test(_tern, nargs=3)
async def infer_type_tern(engine, x, y, z):
    ret_t = await engine.assert_same('type', x, y, z)
    assert isinstance(ret_t, (Int, Float))
    return ret_t


# Coercion

_to_i64 = Primitive('to_i64')


def impl_to_i64(x):
    return int(x)


pyimpl_test[_to_i64] = impl_to_i64


@type_inferrer_test(_to_i64, nargs=1)
async def infer_type_to_i64(engine, x):
    return Int(64)


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

            g = parse(fn)

            print('Args:')
            print(args)

            required_tracks = [main_track]

            inferrer = InferenceEngine(
                g, args,
                tracks={
                    'value': partial(value_track, max_depth=10),
                    'type': partial(type_track)
                },
                required_tracks=required_tracks,
                timeout=0.1
            )

            def out():
                rval = {track: inferrer.output_info(track)
                        for track in required_tracks}
                print('Output of inferrer:')
                print(rval)
                return rval

            print('Expected:')
            print(expected_out)

            try:
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
            finally:
                inferrer.close()

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
        return mul
    m = mul2()
    return m(x, x), m(y, y)


@infer(type=[(i64, InferenceError)])
def test_not_enough_args_prim(x):
    return mul(x)


@infer(type=[(i64, i64, i64, InferenceError)])
def test_too_many_args_prim(x, y, z):
    return mul(x, y, z)


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
            return add
        else:
            return mul

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
            return add
        else:
            return lt

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
            return usub
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
def test_maplist(xs, ys):

    def square(x):
        return x * x

    return maplist(square, xs), maplist(square, ys)


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

    return maplist(adder(z), xs)


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
