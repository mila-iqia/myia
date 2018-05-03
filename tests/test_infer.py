
from pytest import mark

from myia.api import parse
from myia.infer import \
    InferenceEngine, ANYTHING, MyiaTypeError, PrimitiveInferrer, \
    VirtualReference

from myia.dtype import Bool, Int, Float, Tuple as T, List as L
from myia.prim import Primitive
from myia.prim.py_implementations import \
    implementations as pyimpl, \
    add, mul, lt, head, tail
from myia.prim.value_inferrers import \
    ValueTrack, value_inferrer_constructors
from myia.prim.type_inferrers import \
    TypeTrack, type_inferrer_constructors, typeof


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


def fill_in(test, tracks):
    for entry in test:
        if entry is TypeError:
            continue
        for track in tracks:
            if track in entry:
                continue
            elif track == 'type':
                entry[track] = typeof(entry['value'])
            elif track == 'value':
                entry[track] = ANYTHING
    return test


########################
# Temporary primitives #
########################


pyimpl_test = {**pyimpl}
value_inferrer_cons_test = {**value_inferrer_constructors}
type_inferrer_cons_test = {**type_inferrer_constructors}

infer_value_constant = ValueTrack(pyimpl_test, value_inferrer_cons_test)
infer_type_constant = TypeTrack(type_inferrer_cons_test)


def primitive_inferrer(track, op, into):
    def deco(fn):
        def construct(engine):
            return PrimitiveInferrer(engine, track, op, fn)
        into[op] = construct
        return construct

    return deco


# Map

_map = Primitive('map')


def impl_map(f, xs):
    return list(map(f, xs))


pyimpl_test[_map] = impl_map


@primitive_inferrer('type', _map, into=type_inferrer_cons_test)
async def infer_type_map(engine, f, xs):
    f_t = await engine.get('type', f)
    xs_t = await engine.get('type', xs)
    if not isinstance(xs_t, L):
        raise MyiaTypeError('Expect list for map')
    xref = VirtualReference(value=ANYTHING, type=xs_t.element_type)
    ret_t = await f_t(xref)
    return L(ret_t)


# Ternary arithmetic op

_tern = Primitive('tern')


def impl_tern(x, y, z):
    return x + y + z


pyimpl_test[_tern] = impl_tern


@primitive_inferrer('type', _tern, into=type_inferrer_cons_test)
async def infer_type_tern(engine, x, y, z):
    ret_t = await engine.force_same('type', x, y, z)
    assert isinstance(ret_t, (Int, Float))
    return ret_t


# Coercion

_to_i64 = Primitive('to_i64')


def impl_to_i64(x, y, z):
    return x + y + z


pyimpl_test[_to_i64] = impl_to_i64


@primitive_inferrer('type', _to_i64, into=type_inferrer_cons_test)
async def infer_type_to_i64(engine, x):
    return Int(64)


def infer(**tests_spec):

    tests = []
    all_tracks = ['type', 'value']

    for main_track, ts in tests_spec.items():
        if not isinstance(ts, list):
            ts = [ts]
        for t in ts:
            test = []
            for entry in t:
                if entry is TypeError:
                    test.append(TypeError)
                elif isinstance(entry, dict):
                    test.append(entry)
                else:
                    test.append({main_track: entry})
            tests.append(fill_in(test, all_tracks))

    def decorate(fn):
        def run_test(spec):
            *args, expected_out = spec
            g = parse(fn)
            inferrer = InferenceEngine(
                {
                    'value': infer_value_constant,
                    'type': infer_type_constant,
                },
                timeout=0.1
            )
            try:
                out = inferrer.run_sync(g, args)
            except MyiaTypeError as e:
                out = TypeError
            assert out == expected_out

        m = mark.parametrize('spec', list(tests))(run_test)
        m.__orig__ = fn
        return m

    return decorate


type_signature_arith_bin = [
    (i64, i64, i64),
    (f64, f64, f64),
    (i64, f64, TypeError),
    (B, B, TypeError),
]


@infer(type=[({'value': 12.0, 'type': f64},)], value=[(12.0,)])
def test_constants():
    return 1.5 * 8.0


@infer(type=type_signature_arith_bin)
def test_prim_mul(x, y):
    return x * y


@infer(type=[
    (i64, i64, i64, i64),
    (f64, f64, f64, f64),
    (f64, f64, i64, TypeError),
    (i64, f64, f64, TypeError),
    (f64, f64, i64, TypeError),
])
def test_prim_tern(x, y, z):
    return _tern(x, y, z)


@infer(type=[(i64, i64), (f64, f64), (B, TypeError)])
def test_prim_usub(x):
    return -x


@infer(
    type=[
        (B, f64, f64, f64),
        (B, f64, i64, TypeError),
        ({'value': True}, f64, i64, f64),
        ({'value': False}, f64, i64, i64),
        (i64, f64, f64, TypeError),
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
        (f64, f64, TypeError)
    ]
)
def test_while(x, y):
    rval = y
    while x > 0:
        rval = rval * y
        x = x - 1
    return rval


@infer(type=(i64, f64, T(i64, f64)))
def test_tup(x, y):
    return (x, y)


@infer(type=[(T(i64, f64), i64),
             (T(f64, i64), f64),
             (T(), TypeError),
             (f64, TypeError)])
def test_head_tuple(tup):
    return head(tup)


@infer(type=[(T(i64, f64), T(f64)),
             (T(f64, i64), T(i64)),
             (T(), TypeError),
             (f64, TypeError)])
def test_tail_tuple(tup):
    return tail(tup)


@infer(type=[(i64, f64, i64), (f64, i64, f64)])
def test_getitem_tuple(x, y):
    return (x, y)[0]


@infer(
    type=[
        (li64, i64, i64),
        (lf64, i64, f64),
        (lf64, f64, TypeError),
        (f64, i64, TypeError),
        (T(i64, f64), i64, TypeError)
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
        (i64, f64, f64, f64, TypeError),
        (i64, i64, i64, f64, TypeError),
    ]
)
def test_return_closure(w, x, y, z):
    def mul(a):
        def clos(b):
            return a * b
        return clos
    return (mul(w)(x), mul(y)(z))


@infer(type=[(i64, i64), (f64, TypeError)])
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


@infer(type=[(i64, B), (f64, TypeError)])
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
    add, mul

    def choose(i):
        if i == 0:
            return add
        else:
            return mul

    return choose(i)(x, y)


@infer(
    type=[
        (i64, i64, i64, TypeError),
        ({'value': 0}, i64, i64, i64),
        ({'value': 1}, i64, i64, B),
    ]
)
def test_choose_prim_incompatible(i, x, y):
    add, lt

    def choose(i):
        if i == 0:
            return add
        else:
            return lt

    return choose(i)(x, y)


@infer(
    type=[
        (i64, i64, i64, TypeError),
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
        (i64, i64, i64, i64),
        (i64, f64, i64, TypeError)
    ]
)
def test_hof_2(c, x, y):
    _to_i64

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
        (i64, i64, TypeError),
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
        (i64, TypeError)
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


@infer(
    type=[
        ({'value': 4},),
    ]
)
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


@infer(
    type=[
        (li64, lf64, T(li64, lf64)),
        (li64, f64, TypeError),
    ]
)
def test_map(xs, ys):
    def square(x):
        return x * x

    return _map(square, xs), _map(square, ys)


@infer(type=[(i64, TypeError)])
def test_infinite_recursion(x):
    def ouroboros(x):
        return ouroboros(x - 1)

    return ouroboros(x)


@infer(type=[(i64, TypeError)])
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


@infer(type=[(i64, TypeError)])
def test_infinite_mutual_recursion(x):
    return ping()
