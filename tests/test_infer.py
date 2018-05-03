
from pytest import mark

from myia.api import parse
from myia.infer import \
    InferenceEngine, ANYTHING, MyiaTypeError

from myia.dtype import Bool, Int, Float, Tuple as T, List as L
from myia.prim.py_implementations import add, mul, lt, head, tail
from myia.prim.value_inferrers import infer_value_constant
from myia.prim.type_inferrers import infer_type_constant, typeof


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
    (i64, f64, TypeError)
]


@infer(type=type_signature_arith_bin)
def test_prim_mul(x, y):
    return x * y


@infer(type=type_signature_arith_bin)
def test_if(x, y):
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


@infer(type=[(i64, f64, i64), (f64, i64, f64)])
def test_head_tuple(x, y):
    tup = (x, y)
    return head(tup)


@infer(type=[(i64, f64, T(f64)), (f64, i64, T(i64))])
def test_tail_tuple(x, y):
    tup = (x, y)
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
        (i64, T(T(i64, i64), T(B, B)))
    ]
)
def test_hof_2(x):

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
def test_hof_3(x, y):

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
