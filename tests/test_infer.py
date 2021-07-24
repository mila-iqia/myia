import pytest
import operator
from dataclasses import dataclass
from types import SimpleNamespace
from ovld import ovld

import numpy as np

from myia.abstract.data import ANYTHING
from myia.testing import master_placeholders as P
from myia.testing.master_placeholders import (
    J,
    Jinv,
    array_cast,
    array_map,
    array_reduce,
    array_to_scalar,
    bool_and,
    bool_or,
    broadcast_shape,
    conv2d_grad_input,
    dict_values,
    distribute,
    dot,
    embed,
    env_getitem,
    env_setitem,
    gadd,
    grad,
    hastype,
    identity,
    nil_eq,
    nil_ne,
    partial as myia_partial,
    reshape,
    scalar_add,
    scalar_cast,
    scalar_lt,
    scalar_mul,
    scalar_to_array,
    scalar_usub,
    shape,
    switch,
    tagged,
    transpose,
    tuple_setitem,
    typeof,
    unsafe_static_cast,
    user_switch,
    zeros_like,
)
from myia.testing.common import (
    Tuple,
    List,
    B,
    Bot,
    D,
    EmptyTuple,
    Ex,
    Point,
    Point3D,
    S,
    Shp,
    Thing,
    Thing_f,
    Thing_ftup,
    Ty,
    Un as U,
    af16_of,
    af32_of,
    af64_of,
    ai16_of,
    ai32_of,
    ai64_of,
    mysum,
    to_abstract_test,
    Array,
    EnvType as Env,
    External,
    Int,
    Nil,
    Number,
    String,
    f16,
    f32,
    f64,
    i16,
    i32,
    i64,
    u64,
    newenv,
)
from myia.testing.multitest import infer as mt_infer, mt
from myia.abstract.map import MapError as InferenceError
from myia.abstract.map import MapError as MyiaTypeError
ai64 = Array[i64]
af64 = Array[f64]


########################
# Temporary primitives #
########################


def _tern(x: Number, y: Number, z: Number) -> Number:
    return x + y + z


def _to_i64(x: Number) -> Int[64]:
    return int(x)


infer_standard = mt_infer
infer_scalar = mt_infer


type_signature_arith_bin = [
    infer_scalar(i64, i64, result=i64),
    infer_scalar(f64, f64, result=f64),
    infer_scalar(i64, f64, result=InferenceError),
    # Bool type supports arithmetic operations
    infer_scalar(B, B, result=B),
]


@mt(infer_scalar(i64, result=i64), infer_scalar(89, result=89))
def test_identity(x):
    return x


@infer_scalar(result=i64)
def test_constants_int():
    return 2 * 8


@infer_scalar(result=f64)
def test_constants_float():
    return 1.5 * 8.0


@infer_scalar(result=f64)
def test_constants_intxfloat():
    return 8 * 1.5


@infer_scalar(result=f64)
def test_constants_floatxint():
    return 1.5 * 8


@infer_scalar(result=f64)
def test_constants_floatxint2():
    return (8 * 7) + 4.0


@mt(*type_signature_arith_bin)
def test_prim_mul(x, y):
    return x * y


@mt(
    infer_scalar(i64, i64, i64, result=i64),
    infer_scalar(f64, f64, f64, result=f64),
    # Three different inconsistent patterns below
    infer_scalar(f64, f64, i64, result=InferenceError),
    infer_scalar(i64, f64, f64, result=InferenceError),
    # Test too few/too many arguments below
    infer_scalar(i64, result=AssertionError),
    infer_scalar(i64, i64, i64, i64, result=AssertionError),
)
def test_prim_tern(x, y, z):
    return _tern(x, y, z)


@mt(
    infer_scalar(i64, result=i64),
    infer_scalar(f64, result=f64),
    # NB: In Python, -True == -1, -False == 0. So, -bool returns an int.
    # But inference expects result to be same type as input for -x.
    infer_scalar(B, result=B),
)
def test_prim_usub(x):
    return -x


@mt(
    infer_standard(i64, result=f64),
    infer_standard(f32, result=f32),
    infer_standard(f64, result=f64),
    infer_standard(af64_of(2, 5), result=af64_of(2, 5)),
    # NB: Numpy accepts np.log(bool).
    # Maybe inferrer should accept too,
    # by just adding bool type to NUmber union
    infer_standard(B, result=InferenceError),
)
def test_prim_log(x):
    return np.log(x)


@mt(
    infer_scalar(B, f64, f64, result=f64),
    infer_scalar(B, f64, i64, result=InferenceError),
    infer_scalar(True, f64, i64, result=f64),
    infer_scalar(False, f64, i64, result=i64),
    # Note: scalar_pipeline will not convert i64 to bool,
    # so the following is an InferenceError even though it
    # will work with the standard_pipeline
    infer_scalar(i64, f64, f64, result=InferenceError),
    infer_scalar(True, 7, 4, result=i64),
    infer_scalar(False, 7, 4, result=i64),
    infer_scalar(B, 7, 4, result=i64),
)
def test_if(c, x, y):
    if c:
        return x * x
    else:
        return y * y


@mt(*type_signature_arith_bin)
def test_if2(x, y):
    if x > y:
        return x
    else:
        return y


@mt(
    infer_scalar(i64, i64, result=i64),
    infer_scalar(i64, f64, result=f64),
    infer_scalar(f64, f64, result=f64),
    infer_scalar(1_000_000, 3, result=i64),
)
def test_while(x, y):
    rval = y
    while x > 0:
        rval = rval * y
        x = x - 1
    return rval


@mt(
    infer_standard([i64], i64, result=i64),
    infer_standard([i64], f64, result=InferenceError),
    infer_standard(i64, i64, result=InferenceError),
    infer_standard((i64, i64, i64), i64, result=i64),
    infer_standard((i64, f64, i64), i64, result=InferenceError),
)
def test_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


@infer_scalar(i64, f64, result=(i64, f64))
def test_nullary_closure(x, y):
    def make(z):
        def inner():
            return z

        return inner

    a = make(x)
    b = make(y)
    return a(), b()


@infer_scalar(i64, f64, result=(i64, f64))
def test_merge_point(x, y):
    def mul2():
        return scalar_mul

    m = mul2()
    return m(x, x), m(y, y)


@infer_scalar(i64, result=InferenceError)
def test_not_enough_args_prim(x):
    return scalar_mul(x)


@infer_scalar(i64, i64, i64, result=InferenceError)
def test_too_many_args_prim(x, y, z):
    return scalar_mul(x, y, z)


@infer_scalar(i64, result=InferenceError)
def test_not_enough_args(x):
    def g(x, y):
        return x * y

    return g(x)


@infer_scalar(i64, i64, result=InferenceError)
def test_too_many_args(x, y):
    def g(x):
        return x * x

    return g(x, y)


@mt(
    infer_scalar(i64, f64, result=(i64, f64)),
    infer_scalar((i64, i64), f64, result=((i64, i64), f64)),
)
def test_tup(x, y):
    return (x, y)


@mt(
    infer_scalar(i64, i64, result=[i64]),
    infer_scalar(i64, f64, result=InferenceError),
    infer_scalar([i64], [i64], result=[[i64]]),
    # infer_scalar([ai64_of(8, 3)], [ai64_of(4, 3)],
    #              result=[[ai64_of(ANYTHING, 3)]]),
    infer_scalar(ai64_of(4, 7), ai64_of(4, 7), result=[ai64_of(4, 7)]),
    infer_scalar(ai64_of(4, 7), ai64_of(9, 7), result=[ai64_of(ANYTHING, 7)]),
)
def test_list(x, y):
    return [x, y]


@mt(
    infer_scalar(i64, i64, result=[i64]),
    infer_scalar(f64, f64, result=[f64]),
    infer_scalar([f64], [f64], result=InferenceError),
    infer_scalar(i64, f64, result=InferenceError),
)
def test_list_and_scalar(x, y):
    return [x, y, 3]


@infer_scalar(result=[])
def test_list_empty():
    return []


@mt(infer_scalar(1, result=D(x=1)), infer_scalar(f32, result=D(x=f32)))
def test_dict(x):
    return {"x": x}


@infer_scalar(i64, f32, result=D(x=i64, y=f32))
def test_dict2(x, y):
    return {"x": x, "y": y}


@infer_scalar(i64, i64, f32, result=D(x=i64, y=f32))
def test_dict_merge(c, x, y):
    if c == 0:
        return {"x": 1, "y": 2}
    elif c == 1:
        return {"x": 2, "y": 4}
    else:
        return {"x": x, "y": y}


@infer_scalar(i64, f32, result=(i64, f32))
def test_dict_values(x, y):
    return dict_values({"x": x, "y": y})


@infer_scalar(B, i64, f32, result=MyiaTypeError)
def test_dict_incompatible(c, x, y):
    if c:
        return {"x": x, "y": y}
    else:
        return {"x": x, "yy": y}


@mt(
    infer_scalar((), result=0),
    infer_scalar((1,), result=1),
    infer_scalar((i64, f64), result=2),
    infer_scalar([f64], result=i64),
    infer_scalar(af64_of(12, 5), result=np.uint64(12)),
    infer_scalar(af64_of(), result=InferenceError),
    infer_scalar(i64, result=InferenceError),
)
def test_len(xs):
    return len(xs)


@mt(infer_scalar(i64, f64, result=i64), infer_scalar(f64, i64, result=f64))
def test_tuple_getitem(x, y):
    return (x, y)[0]


@mt(infer_scalar(i64, f64, result=f64), infer_scalar(f64, i64, result=i64))
def test_tuple_getitem_negative(x, y):
    return (x, y)[-1]


@infer_scalar(i64, f64, result=InferenceError)
def test_tuple_outofbound(x, y):
    return (x, y)[2]


@mt(
    infer_standard((i64, f64), result=(f64,)),
    infer_standard((f64, i64), result=(i64,)),
    infer_standard((f64, (i64, f64)), result=((i64, f64),)),
    infer_standard((), result=()),
    infer_standard(f64, result=InferenceError),
)
def test_tuple_getslice(tup):
    return tup[1:]


@mt(
    infer_standard((i64, f64, i64), result=(f64,)),
    infer_standard((f64,), result=()),
)
def test_tuple_getslice_2(tup):
    return tup[1:-1]


@mt(
    infer_standard((i64, i64), (i64,), result=(i64, i64, i64)),
    infer_standard((i64, i64), i64, result=InferenceError),
)
def test_concat_tuple(x, y):
    return x + y


@infer_scalar(i64, f64, result=InferenceError)
def test_tuple_outofbound_negative(x, y):
    return (x, y)[-3]


@mt(
    infer_standard(D(x=i64), result=i64),
    infer_standard(D(y=f32), result=InferenceError),
)
def test_dict_getitem(d):
    return d["x"]


@mt(
    infer_standard(D(x=i64), Ex(ANYTHING, t=str), result=InferenceError),
    infer_standard(D(x=i64), 2, result=InferenceError),
)
def test_dict_getitem_nonconst(d, i):
    return d[i]


@mt(
    infer_scalar(D(x=i64), f64, result=D(x=f64)),
    infer_scalar(D(x=i64, y=f32), f64, result=D(x=f64, y=f32)),
    infer_scalar(D(z=i64), f64, result=InferenceError),
)
def test_dict_setitem(d, x):
    return P.dict_setitem(d, "x", x)


@mt(
    infer_scalar((i64, i64), 1, f64, result=(i64, f64)),
    infer_scalar((i64, i64, f64), 1, f64, result=(i64, f64, f64)),
    infer_scalar((i64,), 1, f64, result=InferenceError),
    infer_scalar((i64,), 0.0, f64, result=InferenceError),
    infer_scalar((i64,), i64, f64, result=InferenceError),
)
def test_tuple_setitem(xs, idx, x):
    return tuple_setitem(xs, idx, x)


@infer_scalar(i64, f64, result=(i64, f64))
def test_multitype_function(x, y):
    def mul(a, b):
        return a * b

    return (mul(x, x), mul(y, y))


@mt(*type_signature_arith_bin)
def test_closure(x, y):
    def mul(a):
        return a * x

    return mul(x) + mul(y)


@mt(
    infer_scalar(i64, i64, i64, i64, result=(i64, i64)),
    infer_scalar(f64, f64, f64, f64, result=(f64, f64)),
    infer_scalar(i64, i64, f64, f64, result=(i64, f64)),
    infer_scalar(i64, f64, f64, f64, result=InferenceError),
    infer_scalar(i64, i64, i64, f64, result=InferenceError),
)
def test_return_closure(w, x, y, z):
    def mul(a):
        def clos(b):
            return a * b

        return clos

    return (mul(w)(x), mul(y)(z))


@mt(
    infer_scalar(i64, result=i64),
    infer_scalar(f64, result=f64),
    infer_scalar(i64, i64, result=i64),
    infer_scalar(result=InferenceError),
    infer_scalar(i64, i64, i64, result=InferenceError),
)
def test_default_arg(x, y=3):
    return x + y


@infer_scalar(i64, i64, result=i64)
def test_default_closure(x, y):
    def clos(z=y + y, q=x + x):
        return x + z

    return clos(y)


@infer_standard(result=1)
def test_closure_manager_bug():
    rval = 0
    for z in (1, 2, 3, 4):
        if z == 1:
            rval = z
    return rval


@mt(
    infer_standard(result=0),
    infer_standard(i64, i64, result=i64),
    infer_standard(i64, i64, i64, i64, i64, i64, result=i64),
)
def test_varargs(*args):
    rval = 0
    for arg in args:
        rval = rval + arg
    return rval


@infer_scalar(i64, i64, result=i64)
def test_keywords(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, beatrice=y) + fn(beatrice=3, albert=7)


@infer_scalar(i64, i64, result=i64)
def test_keywords_expand(x, y):
    def fn(z, albert, beatrice):
        return albert - beatrice + z

    return fn(4, **{"albert": x, "beatrice": y})


@infer_scalar(i64, i64, result=InferenceError)
def test_keywords_bad(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, charles=y)


@infer_scalar(i64, i64, result=i64)
def test_keywords_different_order(x, y):
    def fn1(x, albert, beatrice):
        return albert * (x - beatrice)

    def fn2(y, beatrice, albert):
        return y * (albert - beatrice)

    fn = fn1 if x < 0 else fn2

    return fn(5, albert=x, beatrice=y)


@infer_scalar(i64, i64, result=i64)
def test_keywords_defaults(x, y):
    def fn(charles, *, albert=1, beatrice=10):
        return albert - beatrice + charles

    return fn(x, beatrice=y)


@infer_scalar(i64, i64, result=i64)
def test_keywords_shadow(x, y):
    # It used to be that the beatrice arg would be renamed barbara
    # because of the assignment.
    def fn(albert, beatrice):
        barbara = beatrice
        return albert - barbara

    return fn(albert=x, beatrice=y)


@infer_scalar(i64, i64, result=InferenceError)
def test_redundant_kw(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, **{"albert": y, "beatrice": y})


@infer_scalar(i64, result=i64)
def test_defaults_recursive(x):
    def fact(n=x):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    return fact()


@infer_scalar(i64, i64, result=(i64, i64, i64))
def test_kwarg(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(x, beatrice=y), proxy(x, y), proxy(beatrice=x, albert=y)


@infer_scalar(i64, i64, result=InferenceError)
def test_kwarg_bad(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(albert=x, beatrice=y, charles=x + y)


@infer_scalar(i64, i64, result=InferenceError)
def test_keywords_bad_3(x, y):
    return scalar_add(x=x, y=y)


@mt(
    infer_scalar((i64, i64, i64), result=i64),
    infer_scalar((i64, i64, f64), result=InferenceError),
    infer_scalar((i64, i64, i64, i64), result=InferenceError),
    infer_scalar((i64, i64), result=InferenceError),
    infer_scalar(i64, result=InferenceError),
)
def test_apply(args):
    def _f(x, y, z):
        return x + y + z

    return _f(*args)


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


@mt(infer_scalar(i64, result=B), infer_scalar(f64, result=B))
def test_even_odd(n):
    return even(n)


@mt(
    infer_scalar(i64, i64, i64, result=i64),
    infer_scalar(i64, f64, f64, result=f64),
)
def test_choose_prim(i, x, y):
    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_mul

    return choose(i)(x, y)


@mt(
    infer_scalar(i64, i64, i64, result=InferenceError),
    infer_scalar(0, i64, i64, result=i64),
    infer_scalar(1, i64, i64, result=B),
)
def test_choose_prim_incompatible(i, x, y):
    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_lt

    return choose(i)(x, y)


@mt(
    infer_scalar(i64, i64, i64, result=InferenceError),
    infer_scalar(0, i64, i64, result=i64),
    infer_scalar(1, i64, i64, result=B),
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


@mt(infer_scalar(i64, i64, result=i64), infer_scalar(i64, f64, result=f64))
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


@infer_scalar(i64, result=i64)
def test_hof(x):
    def double(x):
        return x + x

    def square(x):
        return x * x

    def hof(f, tup):
        return f(tup[0]) + f(tup[1])

    return hof(double, (x + 1, x + 2)) + hof(square, (x + 3, x + 4))


@mt(
    infer_scalar(i64, i64, result=i64),
    infer_scalar(i64, f64, result=InferenceError),
    infer_scalar(i64, 3, result=i64),
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


@infer_scalar(i64, result=((i64, i64), (B, B)))
def test_hof_3(x):
    def double(x):
        return x + x

    def is_zero(x):
        return x == 0

    def hof(f, tup):
        return (f(tup[0]), f(tup[1]))

    return (hof(double, (x + 1, x + 2)), hof(is_zero, (x + 3, x + 4)))


@mt(
    infer_scalar(i64, i64, result=InferenceError),
    infer_scalar(-1, i64, result=i64),
    infer_scalar(1, i64, result=(i64, i64)),
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


@mt(
    infer_scalar(B, B, i64, i64, result=i64),
    infer_scalar(B, B, f64, f64, result=InferenceError),
    infer_scalar(True, B, (), i64, result=i64),
    infer_scalar(B, True, f64, f64, result=f64),
    infer_scalar(B, True, i64, f64, result=InferenceError),
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


@infer_scalar(i64, i64, result=i64)
def test_func_arg(x, y):
    def g(func, x, y):
        return func(x, y)

    def h(x, y):
        return x + y

    return g(h, x, y)


@infer_scalar(i64, result=InferenceError)
def test_func_arg3(x):
    def g(func, x):
        z = func + x
        return func(z)

    def h(x):
        return x

    return g(h, x)


@mt(infer_scalar(i64, result=i64), infer_scalar(f64, result=f64))
def test_func_arg4(x):
    def h(x):
        return x

    def g(fn, x):
        return fn(h, x)

    def t(fn, x):
        return fn(x)

    return g(t, x)


@infer_scalar(result=i64)
def test_closure_deep():
    def g(x):
        def h():
            return x * x

        return h

    return g(2)()


@infer_scalar(i64, i64, result=i64)
def test_closure_passing(x, y):
    def adder(x):
        def f(y):
            return x + y

        return f

    a1 = adder(1)
    a2 = adder(2)

    return a1(x) + a2(y)


@mt(infer_scalar(B, result=B), infer_scalar(i64, result=InferenceError))
def test_not(x):
    return not x


@mt(infer_scalar(i64, result=True), infer_scalar(f64, result=False))
def test_hastype_simple(x):
    return hastype(x, i64)


@mt(
    infer_scalar(i64, i64, result=InferenceError),
    infer_scalar(i64, Ty(i64), result=True),
    infer_scalar(f64, Ty(i64), result=False),
    infer_scalar((i64, i64), Ty(tuple), result=True),
    infer_scalar((i64, i64), Ty(Tuple), result=True),
    infer_scalar((i64, i64), Ty(Tuple[Number, Number]), result=True),
    infer_scalar((i64, i64), Ty(Tuple[i64, i64]), result=True),
    infer_scalar((i64, i64), Ty(Tuple[float, float]), result=False),
    infer_scalar((i64, i64), Ty(ANYTHING), result=InferenceError),
    infer_scalar([i64], Ty(List), result=True),
    infer_scalar(None, Ty(Nil), result=True),
    infer_scalar(U(i32, i64), Ty(i64), result=B),
    infer_scalar(i32, Ty(U(i16, i32)), result=True),
    infer_scalar(U(i32, i64), Ty(U(i16, i32)), result=B),
)
def test_hastype(x, y):
    return hastype(x, y)


@mt(
    infer_scalar(i64, result=Ty(to_abstract_test(i64))),
    infer_scalar(f64, result=Ty(to_abstract_test(f64))),
)
def test_typeof(x):
    return typeof(x)


Tf4 = Tuple[f64, f64, f64, f64]


@mt(
    infer_standard(i64, result=i64),
    infer_standard(f64, result=i64),
    infer_standard(ai64_of(2, 5), result=0.0),
    infer_standard(af64_of(2, 5), result=0),
    infer_standard((i64, i64), result=i64),
    infer_standard((f64, f64, i64, i64), result=i64),
    infer_standard((f64, f64, f64, f64), result=(f64, f64, f64, f64)),
    infer_standard((i64, (f64, i64)), result=i64),
    infer_standard([i64], result=1.0),
    infer_standard((i64, [i64]), result=i64),
    infer_standard(Point(i64, i64), result=i64),
    infer_standard(Point3D(i64, i64, i64), result=0),
    infer_standard(Thing_ftup, result=(f64, f64)),
    infer_standard(Thing_f, result=0),
    infer_standard(5, result=5),
    infer_standard(Point3D(5, 7, 9), result=0),
    infer_standard(U(ai64_of(2, 5), [f32]), result=f64),
    infer_standard(U([i64], [f32]), result=1.0),
)
def test_hastype_2(x):
    def f(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return f(_to_i64(x))
        elif hastype(x, ai64):
            return 0.0
        elif hastype(x, Point):
            return f(x.x) * f(x.y)
        elif hastype(x, EmptyTuple):
            return 0
        elif hastype(x, Tf4):
            return x
        elif hastype(x, Tuple):
            return f(x[0]) + f(x[1:])
        elif hastype(x, List):
            return 1.0
        elif hastype(x, Thing_ftup):
            return x.contents
        else:
            return 0

    return f(x)


@mt(
    infer_standard(i64, result=i64),
    infer_standard(f64, result=f64),
    infer_standard((i64, i64), result=i64),
    infer_standard((i64, f64), result=InferenceError),
    infer_standard([f64], result=f64),
    infer_standard(Point(i64, i64), result=i64),
)
def test_isinstance(x):
    def f(x):
        if isinstance(x, (int, float)):
            return x
        elif isinstance(x, tuple):
            if len(x) == 0:
                return 0
            else:
                return f(x[0]) + f(x[1:])
        elif isinstance(x, list):
            if x:
                return f(x[0]) + f(x[-1])
            else:
                return 0
        elif isinstance(x, Point):
            return f(x.x) * f(x.y)
        else:
            return None

    return f(x)


@infer_standard(i64, result=InferenceError)
def test_isinstance_bad(x):
    return isinstance(x, (int, 3))


@mt(
    infer_standard(U(i64, (i64, i64)), result=i64),
    infer_standard(U(i64, (f64, i64)), result=InferenceError),
    infer_standard(U(i64, f64), result=InferenceError),
)
def test_union(x):
    if hastype(x, i64):
        return x
    else:
        return x[0]


@mt(
    infer_standard(U(i64, (i64, i64)), result=i64),
)
def test_union_2(x):
    if hastype(x, i64) and x > 0:
        return x
    else:
        return -1


@mt(infer_standard(U(i64, None), result=i64), infer_standard(None, result=0))
def test_union_nil(x):
    if x is None:
        return 0
    else:
        return x


@mt(infer_standard(U(i64, None), U(i64, None), U(i64, None), result=i64))
def test_union_and(x, y, z):
    if (x is not None) and (y is not None) and (z is not None):
        return x + y + z
    else:
        return 0


@infer_standard(U(i64, None), U(i64, None), result=i64)
def test_union_binand(x, y):
    if (x is not None) & (y is not None):
        return x + y
    else:
        return 0


@mt(
    infer_standard(U(i64, f64, (i64, i64)), i64, result=i64),
    infer_standard(U(i64, f64, (i64, i64)), f64, result=InferenceError),
    infer_standard(U(i64, (i64, i64)), f64, result=i64),
    infer_standard(f64, f64, result=f64),
)
def test_union_nested(x, y):
    if hastype(x, i64):
        return x
    elif hastype(x, f64):
        return y
    else:
        return x[0]


@mt(
    infer_standard(U(i64, f64, (i64, i64)), result=i64),
    infer_standard(U(i64, (i64, i64)), result=i64),
)
def test_union_nested_2(x):
    if hastype(x, i64):
        return x
    elif hastype(x, f64):
        return 1234
    else:
        return x[0]


def _square(x):
    return x * x


@infer_scalar(result=InferenceError)
def test_nonexistent_variable():
    return xxxx + yz  # noqa


class helpers:
    add = operator.add
    mul = operator.mul
    square = _square


class data:
    a25 = np.ones((2, 5))


@mt(infer_scalar(i64, result=False), infer_scalar(Point(i64, i64), result=True))
def test_hasattr(x):
    return hasattr(x, "x")


@mt(
    infer_standard(i64, result=i64),
    infer_standard(Point(i64, i64), result=i64),
    infer_standard(U(i64, Point(i64, i64)), result=i64),
)
def test_hasattr_cond(x):
    if hasattr(x, "x"):
        return x.x
    else:
        return x


@mt(
    infer_scalar(i64, i64, result=(i64, i64)),
    infer_scalar(i64, f64, result=InferenceError),
)
def test_getattr(x, y):
    a = helpers.add(x, y)
    b = helpers.mul(x, y)
    c = helpers.square(b)
    return a, c


@mt(
    infer_scalar(i64, i64, result=(i64, i64)),
    infer_scalar(i64, f64, result=(i64, f64)),
)
def test_getattr_multitype(x, y):
    a = helpers.add(x, x)
    b = helpers.add(y, y)
    return a, b


@infer_scalar(result=af64_of(2, 5))
def test_getattr_shape():
    return data.a25


@dataclass
class C1:
    value: object

    def f(self, x):
        return x + self.value


@dataclass
class C2:
    value: object

    def f(self, x):
        return x * self.value


@infer_scalar(U(C1(2), C2(5)), i64, result=i64)
def test_getattr_union(c, x):
    return c.f(x)


_getattr = getattr


@mt(
    infer_scalar("add", i64, result=i64),
    infer_scalar("bad", i64, result=InferenceError),
    infer_scalar(1234, i64, result=InferenceError),
    infer_scalar(External[str], i64, result=InferenceError),
)
def test_getattr_flex(name, x):
    return _getattr(helpers, name)(x, x)


@infer_scalar(External[SimpleNamespace], Ex("surprise"), result=InferenceError)
def test_unknown_data(data, field):
    return _getattr(data, field)


@mt(infer_scalar(i64, i64, result=i64), infer_scalar(f64, f64, result=f64))
def test_method(x, y):
    return x.__add__(y)


@infer_scalar(i64, i64, result=InferenceError)
def test_unknown_method(x, y):
    return x.unknown(y)


@infer_scalar(i64, result=InferenceError)
def test_infinite_recursion(x):
    def ouroboros(x):
        return ouroboros(x - 1)

    return ouroboros(x)


@infer_scalar(i64, result=InferenceError)
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


@infer_scalar(i64, result=InferenceError)
def test_infinite_mutual_recursion(x):
    return ping()


@infer_scalar([i64], result=InferenceError)
def test_recursive_build(xs):
    rval = ()
    for x in xs:
        rval = (x, rval)
    return rval


@mt(
    infer_scalar(af16_of(2, 3), result=Shp(2, 3)),
    infer_scalar(af16_of(2, ANYTHING), result=(S(2, u64), u64)),
)
def test_shape(ary):
    return shape(ary)


@mt(
    infer_scalar(af64_of(2, 3), af64_of(3, 4), result=af64_of(2, 4)),
    infer_scalar(af64_of(2, 3), af32_of(3, 4), result=InferenceError),
    infer_scalar(af64_of(2), af64_of(3, 4), result=InferenceError),
    infer_scalar(af64_of(2, 2), af64_of(3, 4), result=InferenceError),
)
def test_dot(a, b):
    return dot(a, b)


@mt(
    infer_scalar(ai32_of(4), Shp(2, 4), result=ai32_of(2, 4)),
    infer_scalar(ai32_of(4), (u64, u64), result=ai32_of(ANYTHING, ANYTHING)),
    infer_scalar(ai32_of(4), Shp(5, 2), result=InferenceError),
    infer_scalar(ai32_of(4, 2), Shp(4), result=InferenceError),
    infer_scalar(ai32_of(1), Shp(4), result=ai32_of(4)),
    infer_scalar(i32, Shp(4), result=InferenceError),
    infer_scalar([i32], Shp(4), result=InferenceError),
)
def test_distribute(v, shp):
    return distribute(v, shp)


@mt(
    infer_scalar(ai32_of(3, 7), result=ai32_of(3, 7)),
    infer_scalar(ai32_of(7), result=ai32_of(3, 7)),
    infer_scalar(ai32_of(1), result=ai32_of(3, 7)),
    infer_scalar(ai32_of(1, 7), result=ai32_of(3, 7)),
    infer_scalar(ai32_of(3), result=InferenceError),
)
def test_distribute2(v):
    return distribute(v, (3, 7))


@mt(
    infer_scalar(af16_of(1, 2, 3), Shp(6), result=af16_of(6)),
    infer_scalar(af16_of(1, 2, 3), (u64,), result=af16_of(ANYTHING)),
    infer_scalar(af16_of(2, 3), Shp(7), result=InferenceError),
)
def test_reshape(v, shp):
    return reshape(v, shp)


@mt(
    infer_scalar(af16_of(6, 7), Shp(0, 1), result=af16_of(6, 7)),
    infer_scalar(af16_of(6, 7), Shp(1, 0), result=af16_of(7, 6)),
    infer_scalar(af16_of(3, 4, 5), Shp(2, 0, 1), result=af16_of(5, 3, 4)),
    infer_scalar(
        af16_of(3, 4, 5),
        (u64, u64, u64),
        result=af16_of(ANYTHING, ANYTHING, ANYTHING),
    ),
    infer_scalar(af16_of(3, 4, 5), (i64, i64, i64), result=InferenceError),
    infer_scalar(af16_of(3, 4, 5), Shp(1, 0), result=InferenceError),
    infer_scalar(af16_of(3, 4, 5), Shp(1, 2, 9), result=InferenceError),
)
def test_transpose(v, perm):
    return transpose(v, perm)


@mt(
    infer_scalar(af16_of(6, 7), result=af16_of(7, 6)),
    infer_scalar(af16_of(6, 7, 8), result=af16_of(8, 7, 6)),
)
def test_transpose_method(v):
    return v.T


@mt(
    infer_scalar(af16_of(6, 7), result=2),
    infer_scalar(af16_of(6, 7, 8), result=3),
)
def test_ndim(v):
    return v.ndim


@mt(
    infer_scalar(af32_of(3, 4), result=af32_of(3, 4)),
    infer_scalar(ai64_of(3, 4, 5), result=ai64_of(3, 4, 5)),
    infer_scalar(i64, result=InferenceError),
)
def test_array_map(ary):
    def f(v):
        return v + 1

    return array_map(f, ary)


@mt(
    infer_scalar(af32_of(3, 4), af32_of(3, 4), result=af32_of(3, 4)),
    infer_scalar(af32_of(3, 4), af32_of(3, 7), result=InferenceError),
    infer_scalar(
        ai64_of(3, ANYTHING), ai64_of(ANYTHING, 7), result=ai64_of(3, 7)
    ),
    infer_scalar(
        af32_of(3, ANYTHING),
        af32_of(ANYTHING, ANYTHING),
        result=af32_of(3, ANYTHING),
    ),
    infer_scalar(af32_of(3, 4, 5), af32_of(3, 4), result=InferenceError),
    infer_scalar(i64, af32_of(3, 4), result=InferenceError),
)
def test_array_map2(ary1, ary2):
    def f(v1, v2):
        return v1 + v2

    return array_map(f, ary1, ary2)


@infer_scalar(result=InferenceError)
def test_array_map0():
    def f():
        return 1

    return array_map(f)


@mt(
    infer_scalar(
        af32_of(3, 4), af32_of(3, 4), af32_of(3, 4), result=af32_of(3, 4)
    ),
    infer_scalar(
        af32_of(3, 4), af32_of(3, 4), af32_of(3, 7), result=InferenceError
    ),
    infer_scalar(i64, af32_of(3, 4), af32_of(3, 4), result=InferenceError),
    infer_scalar(af32_of(3, 4), i64, af32_of(3, 4), result=InferenceError),
    infer_scalar(af32_of(3, 4), af32_of(3, 4), i64, result=InferenceError),
    infer_scalar(
        af32_of(3, ANYTHING, 5, 6),
        af32_of(3, 4, 5, ANYTHING),
        af32_of(ANYTHING, ANYTHING, ANYTHING, 6),
        result=af32_of(3, 4, 5, 6),
    ),
    infer_scalar(
        af32_of(3, ANYTHING, 5, 6),
        af32_of(3, 4, 5, ANYTHING),
        af32_of(ANYTHING, ANYTHING, ANYTHING, 7),
        result=InferenceError,
    ),
    infer_scalar(
        af32_of(3, 4, 5), af32_of(3, 4), af32_of(3, 4), result=InferenceError
    ),
)
def test_array_map3(ary1, ary2, ary3):
    def f(v1, v2, v3):
        return v1 + v2 + v3

    return array_map(f, ary1, ary2, ary3)


@mt(
    infer_scalar(ai64_of(3, 4), Shp(3, 4), result=ai64_of(3, 4)),
    # infer_scalar(ai64_of(3, 4), Shp(3, ANYTHING),
    #              result=ai64_of(3, ANYTHING)),
    infer_scalar(ai64_of(3, 4), Shp(3, 1), result=ai64_of(3, 1)),
    infer_scalar(ai64_of(3, 4), Shp(1, 4), result=ai64_of(1, 4)),
    infer_scalar(ai64_of(3, 4), Shp(3, 1, 1), result=InferenceError),
    infer_scalar(ai64_of(3, 4), Shp(4, 1), result=InferenceError),
    infer_scalar(ai64_of(3, 4), Shp(4), result=ai64_of(4)),
    infer_scalar(ai64_of(3, 4), Shp(1), result=ai64_of(1)),
    infer_scalar(ai64_of(3, 4), Shp(), result=ai64_of()),
    infer_scalar(i64, Shp(3, 4), result=InferenceError),
)
def test_array_reduce(ary, shp):
    def f(a, b):
        return a + b

    return array_reduce(f, ary, shp)


@infer_scalar(i64, result=i64)
def test_partial_1(x):
    def f(a, b):
        return a + b

    f2 = myia_partial(f, 2)
    return f2(x)


@infer_scalar(i64, result=i64)
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


@mt(
    infer_scalar(i64, result=i64),
    infer_scalar(ai64_of(6, 13), result=ai64_of(6, 13)),
)
def test_identity_function(x):
    return identity(x)


@mt(
    infer_scalar(B, B, result=B),
    infer_scalar(i64, B, result=InferenceError),
    infer_scalar(B, i64, result=InferenceError),
)
def test_bool_and(x, y):
    return bool_and(x, y)


@mt(
    infer_scalar(B, B, result=B),
    infer_scalar(i64, B, result=InferenceError),
    infer_scalar(B, i64, result=InferenceError),
)
def test_bool_or(x, y):
    return bool_or(x, y)


@mt(infer_standard(B, result=False), infer_standard(None, result=True))
def test_nil_eq(x):
    return nil_eq(None, x)


@mt(infer_standard(B, result=True), infer_standard(None, result=False))
def test_nil_ne(x):
    return nil_ne(None, x)


@infer_standard(i64, result=0)
def test_bool_ne(x):
    if None:
        return x
    else:
        return 0


@mt(infer_scalar(B, B, result=B), infer_scalar(i64, i64, result=InferenceError))
def test_and(x, y):
    return x and y


@mt(infer_scalar(i64, None, result=i64), infer_scalar(i64, i64, result=i64))
def test_and_none(x, y):
    if x > 0 and y is not None:
        return x + y
    else:
        return x


@mt(
    infer_scalar(B, i64, i64, result=i64),
    infer_scalar(i64, i64, i64, result=InferenceError),
    infer_scalar(B, i64, f64, result=InferenceError),
    infer_scalar(True, i64, f64, result=i64),
    infer_scalar(False, i64, f64, result=f64),
    infer_scalar(True, 1, 2, result=1),
    infer_scalar(False, 1, 2, result=2),
    infer_scalar(B, 1, 2, result=i64),
    infer_scalar(B, ai64_of(6, 13), ai64_of(6, 13), result=ai64_of(6, 13)),
    infer_scalar(
        B, ai64_of(6, 13), ai64_of(6, 14), result=ai64_of(6, ANYTHING)
    ),
    infer_scalar(True, ai64_of(6, 13), ai64_of(6, 14), result=ai64_of(6, 13)),
    infer_scalar(False, ai64_of(6, 13), ai64_of(6, 14), result=ai64_of(6, 14)),
)
def test_switch(c, x, y):
    return switch(c, x, y)


@infer_scalar(i64, i64, result=i64)
def test_switch_switch(x, y):
    def f1(z):
        return z > 0

    def f2(z):
        return z < 0

    f = switch(x > 0, f1, f2)
    return switch(f(y), 1, 2)


@infer_standard(i64, i64, result=InferenceError)
def test_user_switch_hastype(x, y):
    return user_switch(hastype(x, i64), y + 1, y + 2)


@infer_standard(B, i64, result=i64)
def test_closure_in_data(c, x):
    def f(x):
        return x * x

    def g(x):
        return x + x

    a = Thing((1, [f]))
    b = Thing((2, [g]))
    _, h = switch(c, a, b).contents
    return h[0](x)


@mt(
    infer_scalar(i64, Ty(i64), result=i64),
    infer_scalar(i64, Ty(i16), result=i16),
    infer_scalar(f64, Ty(i16), result=i16),
    infer_scalar(f16, Ty(f32), result=f32),
    infer_scalar(f16, Ty(np.float32), result=f32),
    infer_scalar(f16, Ty(np.dtype("float32")), result=f32),
    infer_scalar(f16, Ty(ANYTHING), result=InferenceError),
    infer_scalar(f16, Ty(Bot), result=InferenceError),
    infer_scalar(f16, Ty(B), result=InferenceError),
    infer_scalar(B, Ty(f32), result=InferenceError),
)
def test_scalar_cast(x, t):
    return scalar_cast(x, t)


@infer_scalar(i64, result=f32)
def test_scalar_cast_2(x):
    return scalar_cast(x, np.float32)


_npf32 = np.dtype("float32")


@infer_scalar(i64, result=f32)
def test_scalar_cast_3(x):
    return scalar_cast(x, _npf32)


@mt(
    infer_scalar(
        ai64_of(2, 3), Ty(to_abstract_test(i64)), result=ai64_of(2, 3)
    ),
    infer_scalar(
        ai64_of(2, 3), Ty(to_abstract_test(i16)), result=ai16_of(2, 3)
    ),
    infer_scalar(
        af64_of(2, 3), Ty(to_abstract_test(i16)), result=ai16_of(2, 3)
    ),
    infer_scalar(
        af16_of(2, 3), Ty(to_abstract_test(f32)), result=af32_of(2, 3)
    ),
    infer_scalar(af16_of(2, 3), Ty(ANYTHING), result=InferenceError),
    infer_scalar(af16_of(2, 3), Ty(Bot), result=InferenceError),
    infer_scalar(af16_of(2, 3), Ty(to_abstract_test(B)), result=InferenceError),
    infer_scalar(B, Ty(to_abstract_test(f32)), result=InferenceError),
)
def test_array_cast(x, t):
    return array_cast(x, t)


@mt(
    infer_scalar(i64, result=ai64_of()),
    infer_scalar(f64, result=af64_of()),
    infer_scalar(af64_of(9, 7), result=InferenceError),
    infer_scalar((i64,), result=InferenceError),
)
def test_scalar_to_array(x):
    return scalar_to_array(x)


@mt(
    infer_scalar(ai64_of(), result=i64),
    infer_scalar(af64_of(), result=f64),
    infer_scalar(af64_of(3, 4), result=InferenceError),
    infer_scalar(af64_of(1, 1, 1), result=InferenceError),
)
def test_array_to_scalar(x):
    return array_to_scalar(x)


@mt(
    infer_scalar((u64,), (u64,), result=(u64,)),
    infer_scalar((u64, u64), (u64,), result=(u64, u64)),
    infer_scalar((u64,), (u64, u64), result=(u64, u64)),
    infer_scalar(Shp(1, 3), Shp(2, 1), result=Shp(2, 3)),
    infer_scalar(Shp(2, 3), Shp(2, 3), result=Shp(2, 3)),
    infer_scalar(Shp(2, 4), Shp(2, 3), result=InferenceError),
    infer_scalar((u64, u64), Shp(2, 3), result=Shp(2, 3)),
    infer_scalar((i64,), (u64,), result=InferenceError),
    infer_scalar((u64,), (i64,), result=InferenceError),
    infer_scalar(i64, i64, result=InferenceError),
)
def test_broadcast_shape(xs, ys):
    return broadcast_shape(xs, ys)


@infer_scalar(i64, i64, result=InferenceError)
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


@mt(
    infer_scalar(i64, i64, i64, result=i64),
    infer_scalar(f64, f64, f64, result=InferenceError),
)
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


@ovld
def mystery(x: ai64, y: ai64):
    return x @ y


@ovld
def mystery(x: af64, y: af64):
    return array_map(scalar_add, x, y)


@mt(
    infer_scalar(ai64_of(7, 9), ai64_of(9, 2), result=ai64_of(7, 2)),
    infer_scalar(af64_of(7, 9), af64_of(9, 2), result=InferenceError),
    infer_scalar(ai64_of(7, 9), ai64_of(7, 9), result=InferenceError),
    infer_scalar(af64_of(7, 9), af64_of(7, 9), result=af64_of(7, 9)),
    infer_scalar(f64, f64, result=InferenceError),
)
def test_multitype_2(x, y):
    return mystery(x, y)


###########################
# Using standard_pipeline #
###########################


@mt(
    infer_standard(i64, i64, result=i64),
    infer_standard(ai64_of(7, 9), i64, result=InferenceError),
)
def test_max_std(x, y):
    if x > y:
        return x
    else:
        return y


@mt(
    infer_scalar(Point(i64, i64), result=i64),
    infer_scalar(Point(f64, f64), result=f64),
)
def test_class(pt):
    return pt.x + pt.y


@mt(
    infer_scalar(Point(i64, i64), result=i64),
    infer_scalar(Point(f64, f64), result=f64),
)
def test_dataclass_method(pt):
    return pt.abs()


@mt(
    infer_scalar(Point(i64, i64), result=i64),
    infer_scalar(Point(f64, f64), result=f64),
)
def test_dataclass_property(pt):
    return pt.absprop


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_add(pt1, pt2):
    return pt1 + pt2


@infer_standard(Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_add_ct(pt):
    return pt + 10


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_sub(pt1, pt2):
    return pt1 - pt2


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_mul(pt1, pt2):
    return pt1 * pt2


@infer_standard(
    Point3D(f64, f64, f64),
    Point3D(f64, f64, f64),
    result=Point3D(f64, f64, f64),
)
def test_arithmetic_data_truediv(pt1, pt2):
    return pt1 / pt2


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_floordiv(pt1, pt2):
    return pt1 // pt2


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_mod(pt1, pt2):
    return pt1 % pt2


@infer_standard(Point(i64, i64), Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_pow(pt1, pt2):
    return pt1 ** pt2


@infer_standard(Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_pos(pt):
    return +pt


@infer_standard(Point(i64, i64), result=Point(i64, i64))
def test_arithmetic_data_neg(pt):
    return -pt


@mt(
    infer_scalar(i64, i64, i64, i64, result=Point(i64, i64)),
    infer_scalar(f64, f64, f64, f64, result=InferenceError),
)
def test_dataclass_inst(x1, y1, x2, y2):
    pt1 = Point(x1, y1)
    pt2 = Point(x2, y2)
    return Point(pt1.x + pt2.x, pt1.y + pt2.y)


@infer_scalar(i64, i64, i64, result=InferenceError)
def test_dataclass_bad_inst(x, y, z):
    return Point(x, y, z)


@infer_scalar(Ty(ANYTHING), i64, i64, result=InferenceError)
def test_dataclass_bad_inst2(cls, x, y):
    return P.make_record(cls, x, y)


@infer_scalar(Point(i64, i64), result=InferenceError)
def test_dataclass_wrong_field(pt):
    return pt.z


@infer_scalar(Thing(i64), result=i64)
def test_dataclass_call(thing):
    return thing()


@infer_scalar(Thing(i64), f64, result=Thing(f64))
def test_record_setitem(thing, x):
    return P.record_setitem(thing, "contents", x)


@mt(
    infer_scalar(Point(i64, i64), i64, result=Point(i64, i64)),
    infer_scalar(Point(i64, i64), f64, result=InferenceError),
)
def test_record_setitem_2(pt, x):
    return P.record_setitem(pt, "x", x)


@infer_scalar(Thing(i64), f64, result=InferenceError)
def test_record_setitem_wrong_field(thing, x):
    return P.record_setitem(thing, "shfifty_five", x)


@mt(
    infer_scalar(i64, i64, result=i64),
    infer_scalar(f64, f64, result=f64),
    infer_scalar([f64], [f64], result=[f64]),
    infer_scalar((i64, f64), (i64, f64), result=(i64, f64)),
    infer_scalar(Point(i64, i64), Point(i64, i64), result=Point(i64, i64)),
    infer_scalar(ai64_of(2, 5), ai64_of(2, 5), result=ai64_of(2, 5)),
    infer_scalar(ai64_of(2, 1), ai64_of(1, 5), result=InferenceError),
    infer_scalar(Env, Env, result=Env),
    infer_scalar(
        {"x": i64, "y": i64}, {"x": i64, "y": i64}, result={"x": i64, "y": i64}
    ),
)
def test_gadd(x, y):
    return gadd(x, y)


@mt(
    infer_scalar(i64, result=0),
    infer_scalar(f64, result=0.0),
    infer_scalar([f64], result=[f64]),  # list element types are broadened
    infer_scalar((i64, f64), result=(0, 0.0)),
    infer_scalar(Point(i64, i64), result=Point(0, 0)),
    infer_scalar(ai64_of(2, 5), result=ai64_of(2, 5, value=0)),
    infer_scalar(af32_of(2, 5), result=af32_of(2, 5, value=0)),
    infer_scalar((2, 3.0), result=(0, 0.0)),
    infer_scalar(Point(1, 2), result=Point(0, 0)),
)
def test_zeros_like(x):
    return zeros_like(x)


# TODO: fix test
@infer_scalar(i64, result=newenv)
def test_zeros_like_fn(x):
    def f(y):
        return x + y

    return zeros_like(f)


@mt(
    infer_scalar(i32, i32, i32, result=i32),
    infer_scalar(i32, f32, i32, result=InferenceError),
    infer_scalar(i32, i32, f32, result=InferenceError),
)
def test_env(x, y, z):
    e = newenv
    e = env_setitem(e, embed(x), y)
    return env_getitem(e, embed(x), z)


@infer_scalar(result=Env)
def test_env_onfn():
    def f(x):
        return x * x

    e = newenv
    e = env_setitem(e, embed(f), newenv)
    return env_getitem(e, embed(f), newenv)


_i64 = to_abstract_test(i64)


@mt(
    infer_scalar(i32, result=i64),
    infer_scalar(f64, result=i64),
    infer_scalar((i32, i64), result=i64),
)
def test_unsafe_static_cast(x):
    return unsafe_static_cast(x, _i64)


@mt(
    infer_scalar(i32, i32, result=InferenceError),
    infer_scalar(i32, (i32, i32), result=InferenceError),
)
def test_unsafe_static_cast_error(x, y):
    return unsafe_static_cast(x, y)


@infer_scalar(i32, result=i32)
def test_pass(x):
    if x < 0:
        x = -x
    else:
        pass
    return x


@infer_scalar(i64, i32, i64, result=U(i32, i64))
def test_tagged(x, y, z):
    if x > 0:
        return tagged(y)
    else:
        return tagged(z)


@infer_scalar(i64, i16, i32, i64, result=U(i16, i32, i64))
def test_tagged_more(c, x, y, z):
    if c == 0:
        return tagged(x)
    elif c > 0:
        return tagged(y)
    else:
        return tagged(z)


@infer_scalar(i64, result=InferenceError)
def test_tagged_too_many_arguments(x):
    return tagged(x, 1, 2)


@mt(
    infer_standard(i32, result=i32),
    infer_standard(f64, result=f64),
    infer_standard(ai64_of(4, 5), result=ai64_of(4, 5)),
)
def test_Jinv2(x):
    def f(x):
        return x * x

    ff = Jinv(J(f))
    return ff(x)


@infer_scalar(i32, result=InferenceError)
def test_Jinv3(x):
    def f(x):
        return x * x

    return Jinv(f)(x)


@infer_scalar(i32, result=InferenceError)
def test_Jinv4(x):
    return Jinv(scalar_add)(x)


@infer_standard(af32_of(5, 7), result=(f32, (Env, af32_of(5, 7))))
def test_J_array(xs):
    def prod(xs):
        p = array_reduce(lambda x, y: x * y, xs, ())
        return array_to_scalar(p)

    jy, bprop = J(prod)(J(xs))
    return Jinv(jy), bprop(1.0)


@infer_standard(f64, result=InferenceError)
def test_J_bprop_invalid(x):
    def f(x):
        return x * x

    _, bprop = J(f)(J(x))
    return bprop(1.0, 1.0)


@infer_standard(f64, result=(f64, f64))
def test_J_return_function(x):
    def f(y):
        return y * y

    def g():
        return f

    jg, _ = J(g)()
    jy, bprop = jg(J(x))
    _, dy = bprop(1.0)
    return Jinv(jy), dy


@mt(infer_standard(f32, f32, result=f32), infer_standard(i16, i16, result=i16))
def test_grad(x, y):
    def f(x, y):
        return x * (y + x)

    return grad(f)(x, y)


@mt(
    infer_standard(i64, result=i64),
    infer_standard(f32, result=f32),
    infer_standard(f64, result=f64),
)
def test_grad_scalar_cast(x):
    def f(x):
        return scalar_cast(x, f16)

    return grad(f)(x)


@mt(
    infer_standard(ai64_of(2, 3), result=ai64_of(2, 3)),
    infer_standard(af32_of(2, 3), result=af32_of(2, 3)),
    infer_standard(af64_of(2, 3), result=af64_of(2, 3)),
)
def test_grad_array_cast(x):
    def f(x):
        return sum(array_cast(x, f16))

    return grad(f)(x)


@infer_standard(af16_of(2, 5), af16_of(2, 5), result=af16_of(2, 5))
def test_grad_reduce(xs, ys):
    def f(xs, ys):
        return array_reduce(scalar_add, xs * ys, ())

    return grad(f)(xs, ys)


@mt(
    infer_standard(None, f64, result=False),
    infer_standard(B, f64, result=False),
    infer_standard(f64, f64, result=MyiaTypeError),
    infer_standard(B, B, result=B),
    infer_standard(None, None, result=True),
    infer_standard(None, NotImplemented, result=False),
    infer_standard(NotImplemented, NotImplemented, result=True),
)
def test_is(x, y):
    return x is y


@mt(
    infer_standard(None, f64, result=True),
    infer_standard(B, f64, result=True),
    infer_standard(f64, f64, result=MyiaTypeError),
    infer_standard(B, B, result=B),
    infer_standard(None, None, result=False),
    infer_standard(None, NotImplemented, result=True),
    infer_standard(NotImplemented, NotImplemented, result=False),
)
def test_is_not(x, y):
    return x is not y


@mt(
    infer_scalar(
        af32_of(1, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        u64,
        result=af32_of(1, 3, 2, 1),
    ),
    infer_scalar(
        af32_of(2, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        u64,
        result=af32_of(2, 3, 2, 1),
    ),
    infer_scalar(
        af32_of(2, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        Shp(2, 3, 4),
        Shp(3, 2),
        Shp(3, 4),
        u64,
        result=InferenceError,
    ),
    infer_scalar(
        af32_of(2, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        Shp(2, 3),
        Shp(3, 2, 4),
        Shp(3, 4),
        u64,
        result=InferenceError,
    ),
    infer_scalar(
        af32_of(2, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4, 2),
        u64,
        result=InferenceError,
    ),
)
def test_conv2d(i, w, s, p, d, g):
    return P.conv2d(i, w, s, p, d, g)


@mt(
    infer_scalar(
        Shp(1, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        af32_of(1, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(1, 3, 4, 5),
    ),
    infer_scalar(
        Shp(2, 3, 4, 5),
        af32_of(3, 1, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(2, 3, 4, 5),
    ),
    infer_scalar(
        Shp(2, 6, 4, 5),
        af32_of(3, 2, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(2, 6, 4, 5),
    ),
    infer_scalar(
        Shp(2, 1, 4, 5),
        af32_of(3, 1, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(1, u64),
        result=af32_of(2, 1, 4, 5),
    ),
)
def test_conv2d_grad_input(i_s, w, g_o, s, p, d, g):
    return conv2d_grad_input(i_s, w, g_o, s, p, d, g)


@mt(
    infer_scalar(
        af32_of(1, 3, 4, 5),
        Shp(3, 1, 3, 3),
        af32_of(1, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(3, 1, 3, 3),
    ),
    infer_scalar(
        af32_of(2, 3, 4, 5),
        Shp(3, 1, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(3, 1, 3, 3),
    ),
    infer_scalar(
        af32_of(2, 6, 4, 5),
        Shp(3, 2, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(3, u64),
        result=af32_of(3, 2, 3, 3),
    ),
    infer_scalar(
        af32_of(2, 1, 4, 5),
        Shp(3, 1, 3, 3),
        af32_of(2, 3, 2, 1),
        Shp(2, 3),
        Shp(3, 2),
        Shp(3, 4),
        S(1, u64),
        result=af32_of(3, 1, 3, 3),
    ),
)
def test_conv2d_weight_grad(i, w_s, g_o, s, p, d, g):
    return P.conv2d_weight_grad(i, w_s, g_o, s, p, d, g)


@mt(
    infer_standard("idk", result=i64),
    infer_standard("hey", result=2),
    infer_standard(String, result=i64),
)
def test_string_eq(s):
    x = 2
    if s == "idk":
        x = x + 1
    return x


@mt(
    infer_standard("idk", result=2),
    infer_standard("hey", result=i64),
    infer_standard(String, result=i64),
)
def test_string_ne(s):
    x = 2
    if s != "idk":
        x = x + 1
    return x


@mt(infer_standard("hey", result="hey"), infer_standard(String, result=String))
def test_string_return(s):
    return s
