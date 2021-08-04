import math
import operator
from dataclasses import dataclass
from types import SimpleNamespace
from myia.parser import MyiaSyntaxError

import pytest

from myia.abstract.data import ANYTHING
from myia.abstract.map import (
    MapError as InferenceError,
    MapError as MyiaTypeError,
)
from myia.basics import (
    partial as myia_partial,
    user_switch,
    user_switch as switch,
)
from myia.testing.common import (
    B,
    Bot,
    D,
    EmptyTuple,
    Ex,
    External,
    Int,
    Nil,
    Number,
    Point,
    Point3D,
    String,
    Thing,
    Thing_f,
    Thing_ftup,
    Ty,
    Un as U,
    list_of,
    mysum,
    tuple_of,
)
from myia.testing.master_placeholders import (
    dict_setitem,
    dict_values,
    hastype,
    identity,
    scalar_add,
    scalar_cast,
    scalar_lt,
    scalar_mul,
    scalar_usub,
    tuple_setitem,
    unsafe_static_cast,
    zeros_like,
)
from myia.testing.multitest import infer as mt_infer, mt
from myia.testing.testing_inferrers import add_testing_inferrers

add_testing_inferrers()


def mark_fail(exc_type, reason=None):
    return pytest.mark.xfail(strict=True, raises=exc_type, reason=reason)


def _tern(x: Number, y: Number, z: Number) -> Number:
    return x + y + z


def _to_i64(x: Number) -> Int:
    return int(x)


infer_standard = mt_infer
infer_scalar = mt_infer


type_signature_arith_bin = [
    infer_scalar(int, int, result=int),
    infer_scalar(float, float, result=float),
    infer_scalar(int, float, result=InferenceError),
    # Bool type supports arithmetic operations
    infer_scalar(B, B, result=B),
]

@mt(infer_scalar(int, result=int), infer_scalar(89, result=89))
def test_identity(x):
    return x


@infer_scalar(result=int)
def test_constants_int():
    return 2 * 8


@infer_scalar(result=float)
def test_constants_float():
    return 1.5 * 8.0


@infer_scalar(result=float)
def test_constants_intxfloat():
    return 8 * 1.5


@infer_scalar(result=float)
def test_constants_floatxint():
    return 1.5 * 8


@infer_scalar(result=float)
def test_constants_floatxint2():
    return (8 * 7) + 4.0


@mt(
    infer_scalar(int, int, result=int),
    infer_scalar(float, float, result=float),
    # float.__rmul__ support (float, int) args
    infer_scalar(int, float, result=float),
    # Bool type supports arithmetic operations
    infer_scalar(B, B, result=int),
)
def test_prim_mul(x, y):
    return x * y


@mt(
    infer_scalar(int, int, int, result=int),
    infer_scalar(float, float, float, result=float),
    # Three different inconsistent patterns below
    infer_scalar(float, float, int, result=InferenceError),
    infer_scalar(int, float, float, result=InferenceError),
    # Test too few/too many arguments below
    infer_scalar(int, result=AssertionError),
    infer_scalar(int, int, int, int, result=AssertionError),
)
def test_prim_tern(x, y, z):
    return _tern(x, y, z)


@mt(
    infer_scalar(int, result=int),
    infer_scalar(float, result=float),
    # NB: In Python, -True == -1, -False == 0. So, -bool returns an int.
    infer_scalar(B, result=int),
)
def test_prim_usub(x):
    return -x


@mt(
    infer_standard(int, result=float),
    infer_standard(float, result=float),
    # math.log accepts bool
    infer_standard(B, result=float),
)
def test_math_log(x):
    return math.log(x)


@mt(
    infer_scalar(B, float, float, result=float),
    infer_scalar(B, float, int, result=InferenceError),
    # Both if-branches are computed,
    # so this should raise an error in myia zero code
    infer_scalar(True, float, int, result=InferenceError),
    infer_scalar(False, float, int, result=InferenceError),
    infer_scalar(int, float, float, result=float),
    infer_scalar(True, 7, 4, result=int),
    infer_scalar(False, 7, 4, result=int),
    infer_scalar(B, 7, 4, result=int),
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
    infer_scalar(int, int, result=int),
    infer_scalar(int, float, result=float),
    infer_scalar(float, float, result=float),
    infer_scalar(1_000_000, 3, result=int),
)
def test_while(x, y):
    rval = y
    while x > 0:
        rval = rval * y
        x = x - 1
    return rval


@mt(
    infer_standard([int], int, result=int),
    infer_standard([int], float, result=InferenceError),
    infer_standard(
        int, int, result=TypeError("myia_iter: unexpected input type")
    ),
    infer_standard((int, int, int), int, result=int),
    infer_standard((int, float, int), int, result=InferenceError),
)
def test_for(xs, y):
    rval = y
    for x in xs:
        rval = rval + x
    return rval


@infer_scalar(int, float, result=(int, float))
def test_nullary_closure(x, y):
    def make(z):
        def inner():
            return z

        return inner

    a = make(x)
    b = make(y)
    return a(), b()


@infer_scalar(int, float, result=(int, float))
def test_merge_point(x, y):
    def mul2():
        return scalar_mul

    m = mul2()
    return m(x, x), m(y, y)


@infer_scalar(int, result=AssertionError)
def test_not_enough_args_prim(x):
    return scalar_mul(x)


@infer_scalar(int, int, int, result=AssertionError)
def test_too_many_args_prim(x, y, z):
    return scalar_mul(x, y, z)


@infer_scalar(int, result=AssertionError)
def test_not_enough_args(x):
    def g(x, y):
        return x * y

    return g(x)


@infer_scalar(int, int, result=AssertionError)
def test_too_many_args(x, y):
    def g(x):
        return x * x

    return g(x, y)


@mt(
    infer_scalar(int, float, result=(int, float)),
    infer_scalar((int, int), float, result=((int, int), float)),
)
def test_tup(x, y):
    return (x, y)


@mt(
    infer_scalar(int, int, result=[int]),
    infer_scalar(int, float, result=InferenceError),
    infer_scalar([int], [int], result=[[int]]),
)
def test_list(x, y):
    return [x, y]


@mt(
    infer_scalar(int, int, result=[int]),
    infer_scalar([float], [float], result=InferenceError),
    infer_scalar(int, float, result=InferenceError),
)
def test_list_and_scalar(x, y):
    return [x, y, 3]


@mark_fail(InferenceError, "Cannot merge int and float (pass in master, fail hre)")
@mt(
    infer_scalar(float, float, result=[float]),
)
def test_list_and_scalar_different_types(x, y):
    return [x, y, 3]



@infer_scalar(result=[])
def test_list_empty():
    return []


@mt(infer_scalar(1, result=D(x=1)), infer_scalar(float, result=D(x=float)))
def test_dict(x):
    return {"x": x}


@infer_scalar(int, float, result=D(x=int, y=float))
def test_dict2(x, y):
    return {"x": x, "y": y}


# fail
@infer_scalar(int, int, float, result=D(x=int, y=float))
def test_dict_merge(c, x, y):
    if c == 0:
        return {"x": 1, "y": 2}
    elif c == 1:
        return {"x": 2, "y": 4}
    else:
        return {"x": x, "y": y}


@infer_scalar(int, float, result=(int, float))
def test_dict_values(x, y):
    return dict_values({"x": x, "y": y})


@infer_scalar(B, int, float, result=MyiaTypeError)
def test_dict_incompatible(c, x, y):
    if c:
        return {"x": x, "y": y}
    else:
        return {"x": x, "yy": y}


@mt(
    infer_scalar((), result=0),
    infer_scalar((1,), result=1),
    infer_scalar((int, float), result=2),
    infer_scalar([float], result=int),
    infer_scalar(
        int, result=AttributeError("Interface has no attribute __len__")
    ),
)
def test_len(xs):
    return len(xs)


@mt(
    infer_scalar(int, float, result=int), infer_scalar(float, int, result=float)
)
def test_tuple_getitem(x, y):
    return (x, y)[0]


@mt(
    infer_scalar(int, float, result=float), infer_scalar(float, int, result=int)
)
def test_tuple_getitem_negative(x, y):
    return (x, y)[-1]


@infer_scalar(int, float, result=IndexError)
def test_tuple_outofbound(x, y):
    return (x, y)[2]


@mt(
    infer_standard((int, float), result=(float,)),
    infer_standard((float, int), result=(int,)),
    infer_standard((float, (int, float)), result=((int, float),)),
    infer_standard((), result=()),
    infer_standard(float, result=AssertionError(r"getitem can currently only be used for dicts, lists and tuples, got \*float\(\)")),
)
def test_tuple_getslice(tup):
    return tup[1:]


@mt(
    infer_standard((int, float, int), result=(float,)),
    infer_standard((float,), result=()),
)
def test_tuple_getslice_2(tup):
    return tup[1:-1]


@mt(
    infer_standard((int, int), (int,), result=(int, int, int)),
    infer_standard(
        (int, int), int, result=AssertionError("Expected abstract tuple")
    ),
)
def test_concat_tuple(x, y):
    return x + y


@infer_scalar(int, float, result=IndexError)
def test_tuple_outofbound_negative(x, y):
    return (x, y)[-3]


@mt(
    infer_standard(D(x=int), result=int),
    infer_standard(D(y=float), result=ValueError("not in list")),
)
def test_dict_getitem(d):
    return d["x"]


@mt(
    infer_standard(
        D(x=int), Ex(ANYTHING, t=str), result=ValueError("not in list")
    ),
    infer_standard(D(x=int), 2, result=ValueError("not in list")),
)
def test_dict_getitem_nonconst(d, i):
    return d[i]


@mt(
    infer_scalar(D(x=int), float, result=D(x=float)),
    infer_scalar(D(x=int, y=float), float, result=D(x=float, y=float)),
    infer_scalar(D(z=int), float, result=ValueError("not in list")),
)
def test_dict_setitem(d, x):
    return dict_setitem(d, "x", x)


@mt(
    infer_scalar((int, int), 1, float, result=(int, float)),
    infer_scalar((int, int, float), 1, float, result=(int, float, float)),
    infer_scalar((int,), 1, float, result=IndexError),
    infer_scalar(
        (int,), 0.0, float, result=AssertionError("Expected int index")
    ),
    infer_scalar(
        (int,), int, float, result=AssertionError("Expected int value")
    ),
)
def test_tuple_setitem(xs, idx, x):
    return tuple_setitem(xs, idx, x)


@infer_scalar(int, float, result=(int, float))
def test_multitype_function(x, y):
    def mul(a, b):
        return a * b

    return (mul(x, x), mul(y, y))


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder() (around make_handle())")
@mt(
    infer_scalar(int, int, result=int),
    infer_scalar(float, float, result=float),
    # fails in master, should pass here because we support float.__rmul__
    infer_scalar(int, float, result=float),
    infer_scalar(B, B, result=int),
)
def test_closure(x, y):
    def mul(a):
        return a * x

    return mul(x) + mul(y)


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@mt(
    infer_scalar(int, int, int, int, result=(int, int)),
    infer_scalar(float, float, float, float, result=(float, float)),
    infer_scalar(int, int, float, float, result=(int, float)),
    # fails in master, should pass here
    infer_scalar(int, float, float, float, result=(float, float)),
    # fails in master, should pass here
    infer_scalar(int, int, int, float, result=(int, float)),
)
def test_return_closure(w, x, y, z):
    def mul(a):
        def clos(b):
            return a * b

        return clos

    return (mul(w)(x), mul(y)(z))


@mark_fail(MyiaSyntaxError, "Parser does not yet support default values on entry function")
@mt(
    infer_scalar(int, result=int),
    infer_scalar(float, result=float),
    infer_scalar(int, int, result=int),
    infer_scalar(result=InferenceError),
    infer_scalar(int, int, int, result=InferenceError),
)
def test_default_arg(x, y=3):
    return x + y


@infer_scalar(int, int, result=int)
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


@pytest.mark.xfail(strict=True, reason="Errors handling varargs")
@mt(
    infer_standard(result=0),
    infer_standard(int, int, result=int),
    infer_standard(int, int, int, int, int, int, result=int),
)
def test_varargs(*args):
    rval = 0
    for arg in args:
        rval = rval + arg
    return rval


@infer_scalar(int, int, result=int)
def test_keywords(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, beatrice=y) + fn(beatrice=3, albert=7)


@infer_scalar(int, int, result=int)
def test_keywords_expand(x, y):
    def fn(z, albert, beatrice):
        return albert - beatrice + z

    return fn(4, **{"albert": x, "beatrice": y})


@infer_scalar(int, int, result=InferenceError)
def test_keywords_bad(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, charles=y)


@infer_scalar(int, int, result=int)
def test_keywords_different_order(x, y):
    def fn1(x, albert, beatrice):
        return albert * (x - beatrice)

    def fn2(y, beatrice, albert):
        return y * (albert - beatrice)

    fn = fn1 if x < 0 else fn2

    return fn(5, albert=x, beatrice=y)


@infer_scalar(int, int, result=int)
def test_keywords_defaults(x, y):
    def fn(charles, *, albert=1, beatrice=10):
        return albert - beatrice + charles

    return fn(x, beatrice=y)


@infer_scalar(int, int, result=int)
def test_keywords_shadow(x, y):
    # It used to be that the beatrice arg would be renamed barbara
    # because of the assignment.
    def fn(albert, beatrice):
        barbara = beatrice
        return albert - barbara

    return fn(albert=x, beatrice=y)


@infer_scalar(int, int, result=InferenceError)
def test_redundant_kw(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, **{"albert": y, "beatrice": y})


@infer_scalar(int, result=int)
def test_defaults_recursive(x):
    def fact(n=x):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    return fact()


@infer_scalar(int, int, result=(int, int, int))
def test_kwarg(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(x, beatrice=y), proxy(x, y), proxy(beatrice=x, albert=y)


@infer_scalar(int, int, result=InferenceError)
def test_kwarg_bad(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(albert=x, beatrice=y, charles=x + y)


@infer_scalar(int, int, result=InferenceError)
def test_keywords_bad_3(x, y):
    return scalar_add(x=x, y=y)


@mark_fail(AssertionError, "Got a data.Placeholder in processus")
@mt(
    infer_scalar((int, int, int), result=int),
    infer_scalar((int, int, float), result=InferenceError),
    infer_scalar((int, int, int, int), result=InferenceError),
    infer_scalar((int, int), result=InferenceError),
    infer_scalar(int, result=InferenceError),
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


@mt(infer_scalar(int, result=B), infer_scalar(float, result=B))
def test_even_odd(n):
    return even(n)


@mark_fail(AssertionError, "Got a data.Placeholder in processus")
@mt(
    infer_scalar(int, int, int, result=int),
    infer_scalar(int, float, float, result=float),
)
def test_choose_prim(i, x, y):
    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_mul

    return choose(i)(x, y)


@mt(
    infer_scalar(int, int, int, result=InferenceError),
    infer_scalar(0, int, int, result=int),
    infer_scalar(1, int, int, result=B),
)
def test_choose_prim_incompatible(i, x, y):
    def choose(i):
        if i == 0:
            return scalar_add
        else:
            return scalar_lt

    return choose(i)(x, y)


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@mt(
    # This should fail with an inference error if tests pass
    infer_scalar(int, int, int, result=None),
    infer_scalar(0, int, int, result=int),
    infer_scalar(1, int, int, result=B),
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


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@mt(infer_scalar(int, int, result=int), infer_scalar(int, float, result=float))
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


@infer_scalar(int, result=int)
def test_hof(x):
    def double(x):
        return x + x

    def square(x):
        return x * x

    def hof(f, tup):
        return f(tup[0]) + f(tup[1])

    return hof(double, (x + 1, x + 2)) + hof(square, (x + 3, x + 4))


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@mt(
    infer_scalar(int, int, result=int),
    # This fails in master. And here?
    infer_scalar(int, float, result=None),
    infer_scalar(int, 3, result=int),
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


@infer_scalar(int, result=((int, int), (B, B)))
def test_hof_3(x):
    def double(x):
        return x + x

    def is_zero(x):
        return x == 0

    def hof(f, tup):
        return (f(tup[0]), f(tup[1]))

    return (hof(double, (x + 1, x + 2)), hof(is_zero, (x + 3, x + 4)))


@mt(
    infer_scalar(int, int, result=InferenceError),
    infer_scalar(-1, int, result=int),
    infer_scalar(1, int, result=(int, int)),
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
    infer_scalar(B, B, int, int, result=int),
    infer_scalar(B, B, float, float, result=InferenceError),
    infer_scalar(True, B, (), int, result=int),
    infer_scalar(B, True, float, float, result=float),
    infer_scalar(B, True, int, float, result=InferenceError),
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


@infer_scalar(int, int, result=int)
def test_func_arg(x, y):
    def g(func, x, y):
        return func(x, y)

    def h(x, y):
        return x + y

    return g(h, x, y)


@infer_scalar(
    int, result=TypeError(r"No inferrer for <slot wrapper '__radd__' of 'int' objects>")
)
def test_func_arg3(x):
    def g(func, x):
        z = func + x
        return func(z)

    def h(x):
        return x

    return g(h, x)


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@mt(infer_scalar(int, result=int), infer_scalar(float, result=float))
def test_func_arg4(x):
    def h(x):
        return x

    def g(fn, x):
        return fn(h, x)

    def t(fn, x):
        return fn(x)

    return g(t, x)


@infer_scalar(result=int)
def test_closure_deep():
    def g(x):
        def h():
            return x * x

        return h

    return g(2)()


@infer_scalar(int, int, result=int)
def test_closure_passing(x, y):
    def adder(x):
        def f(y):
            return x + y

        return f

    a1 = adder(1)
    a2 = adder(2)

    return a1(x) + a2(y)


# Should work even if x is not a bool.
@mt(infer_scalar(B, result=B), infer_scalar(int, result=B))
def test_not(x):
    return not x


@mt(infer_scalar(int, result=True), infer_scalar(float, result=False))
def test_hastype_simple(x):
    return hastype(x, int)


@mt(
    infer_scalar(int, int, result=AssertionError("Expected abstract type")),
    infer_scalar(int, Ty(int), result=True),
    infer_scalar(float, Ty(int), result=False),
    infer_scalar((int, int), Ty(tuple), result=True),
    infer_scalar((int, int), Ty(tuple_of()), result=True),
    infer_scalar((int, int), Ty(tuple_of(Number, Number)), result=True),
    infer_scalar((int, int), Ty(tuple_of(int, int)), result=True),
    infer_scalar((int, int), Ty(tuple_of(float, float)), result=False),
    infer_scalar(
        (int, int),
        Ty(ANYTHING),
        result=AssertionError(
            "Too broad type `object` expected for isinstance"
        ),
    ),
    infer_scalar([int], Ty(list_of()), result=True),
    infer_scalar(None, Ty(Nil), result=True),
    infer_scalar(U(int, int), Ty(int), result=B),
    infer_scalar(int, Ty(U(int, int)), result=True),
    infer_scalar(U(int, int), Ty(U(int, int)), result=B),
)
def test_hastype(x, y):
    return hastype(x, y)


@mt(
    infer_scalar(int, result=Ty(int)),
    infer_scalar(float, result=Ty(float)),
)
def test_typeof(x):
    return type(x)


Tf4 = tuple_of(float, float, float, float)


@mt(
    infer_standard(int, result=int),
    infer_standard(float, result=int),
    infer_standard((int, int), result=int),
    infer_standard((float, float, int, int), result=int),
    infer_standard(
        (float, float, float, float), result=(float, float, float, float)
    ),
    infer_standard((int, (float, int)), result=int),
    infer_standard([int], result=1.0),
    infer_standard((int, [int]), result=int),
    infer_standard(Point(int, int), result=int),
    infer_standard(Point3D(int, int, int), result=0),
    infer_standard(Thing_ftup, result=(float, float)),
    infer_standard(Thing_f, result=0),
    infer_standard(5, result=5),
    infer_standard(Point3D(5, 7, 9), result=0),
    infer_standard(U([int], [float]), result=1.0),
)
def test_hastype_2(x):
    def f(x):
        if hastype(x, int):
            return x
        elif hastype(x, float):
            return f(_to_i64(x))
        elif hastype(x, Point):
            return f(x.x) * f(x.y)
        elif hastype(x, EmptyTuple):
            return 0
        elif hastype(x, Tf4):
            return x
        elif hastype(x, tuple_of()):
            return f(x[0]) + f(x[1:])
        elif hastype(x, list_of()):
            return 1.0
        elif hastype(x, Thing_ftup):
            return x.contents
        else:
            return 0

    return f(x)


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder() (around make_handle())")
@mt(
    infer_standard(int, result=int),
    infer_standard(float, result=float),
    infer_standard((int, int), result=int),
    # this fails in master. And here?
    infer_standard((int, float), result=None),
    infer_standard([float], result=float),
    infer_standard(Point(int, int), result=int),
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


@infer_standard(int, result=AssertionError("Expected abstract type"))
def test_isinstance_bad(x):
    return isinstance(x, (int, 3))


@mark_fail(AssertionError, "getitem can currently only be used for dicts, lists and tuples (union not well handled by inferrer)")
@mt(
    infer_standard(U(int, (int, int)), result=int),
    infer_standard(U(int, (float, int)), result=InferenceError),
    infer_standard(U(int, float), result=InferenceError),
)
def test_union(x):
    if hastype(x, int):
        return x
    else:
        return x[0]


@mark_fail(InferenceError, "Cannot merge InterfaceTrack containing InferenceFunction")
@mt(
    infer_standard(U(int, (int, int)), result=int),
)
def test_union_2(x):
    if hastype(x, int) and x > 0:
        return x
    else:
        return -1


@mark_fail(InferenceError, "Cannot merge int and NoneType")
@mt(infer_standard(U(int, None), result=int), infer_standard(None, result=0))
def test_union_nil(x):
    if x is None:
        return 0
    else:
        return x


@mark_fail(InferenceError, "Cannot merge InterfaceTrack containing InferenceFunction")
@mt(infer_standard(U(int, None), U(int, None), U(int, None), result=int))
def test_union_and(x, y, z):
    if (x is not None) and (y is not None) and (z is not None):
        return x + y + z
    else:
        return 0


@infer_standard(U(int, None), U(int, None), result=int)
def test_union_binand(x, y):
    if (x is not None) & (y is not None):
        return x + y
    else:
        return 0


@mt(
    infer_standard(U(int, float, (int, int)), int, result=int),
    infer_standard(U(int, float, (int, int)), float, result=InferenceError),
    infer_standard(U(int, (int, int)), float, result=int),
    infer_standard(float, float, result=float),
)
def test_union_nested(x, y):
    if hastype(x, int):
        return x
    elif hastype(x, float):
        return y
    else:
        return x[0]


@mt(
    infer_standard(U(int, float, (int, int)), result=int),
    infer_standard(U(int, (int, int)), result=int),
)
def test_union_nested_2(x):
    if hastype(x, int):
        return x
    elif hastype(x, float):
        return 1234
    else:
        return x[0]


def _square(x):
    return x * x


@infer_scalar(result=NameError)
def test_nonexistent_variable():
    return xxxx + yz  # noqa


class helpers:
    add = operator.add
    mul = operator.mul
    square = _square


@mt(infer_scalar(int, result=False), infer_scalar(Point(int, int), result=True))
def test_hasattr(x):
    return hasattr(x, "x")


@mt(
    infer_standard(int, result=int),
    infer_standard(Point(int, int), result=int),
    infer_standard(U(int, Point(int, int)), result=int),
)
def test_hasattr_cond(x):
    if hasattr(x, "x"):
        return x.x
    else:
        return x


@mt(
    infer_scalar(int, int, result=(int, int)),
    infer_scalar(int, float, result=InferenceError),
)
def test_getattr(x, y):
    a = helpers.add(x, y)
    b = helpers.mul(x, y)
    c = helpers.square(b)
    return a, c


@mt(
    infer_scalar(int, int, result=(int, int)),
    infer_scalar(int, float, result=(int, float)),
)
def test_getattr_multitype(x, y):
    a = helpers.add(x, x)
    b = helpers.add(y, y)
    return a, b


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


@mark_fail(
    AttributeError,
    "`AttributeError: 'Named' object has no attribute 'f'` (dataclasses not yet supported)",
)
@infer_scalar(U(C1(2), C2(5)), int, result=int)
def test_getattr_union(c, x):
    return c.f(x)


_getattr = getattr


@mt(
    infer_scalar("add", int, result=int),
    infer_scalar("bad", int, result=InferenceError),
    infer_scalar(1234, int, result=InferenceError),
    infer_scalar(External(str), int, result=InferenceError),
)
def test_getattr_flex(name, x):
    return _getattr(helpers, name)(x, x)


@infer_scalar(External(SimpleNamespace), Ex("surprise"), result=InferenceError)
def test_unknown_data(data, field):
    return _getattr(data, field)


@mt(
    infer_scalar(int, int, result=int), infer_scalar(float, float, result=float)
)
def test_method(x, y):
    return x.__add__(y)


@infer_scalar(
    int,
    int,
    result=AttributeError("type object 'int' has no attribute 'unknown'"),
)
def test_unknown_method(x, y):
    return x.unknown(y)


@infer_scalar(int, result=InferenceError)
def test_infinite_recursion(x):
    def ouroboros(x):
        return ouroboros(x - 1)

    return ouroboros(x)


@infer_scalar(int, result=InferenceError)
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


@infer_scalar(int, result=InferenceError)
def test_infinite_mutual_recursion(x):
    return ping()


# myia_repr_failure prints a too long structure here. Need to deactivate it
# to print readable error trace.
@mark_fail(RecursionError, "maximum recursion depth exceeded while calling a Python object (in `is_concrete[AbstractStructure]`)")
@infer_scalar([int], result=InferenceError)
def test_recursive_build(xs):
    rval = ()
    for x in xs:
        rval = (x, rval)
    return rval


@infer_scalar(int, result=int)
def test_partial_1(x):
    def f(a, b):
        return a + b

    f2 = myia_partial(f, 2)
    return f2(x)


@mark_fail(InferenceError, "Cannot merge *type(?x) and *Placeholder()")
@infer_scalar(int, result=int)
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
    infer_scalar(int, result=int),
)
def test_identity_function(x):
    return identity(x)


@mt(
    infer_scalar(B, B, result=B),
    infer_scalar(int, B, result=InferenceError),
    infer_scalar(B, int, result=InferenceError),
)
def test_bool_and(x, y):
    return x and y


@mt(
    infer_scalar(B, B, result=B),
    infer_scalar(int, B, result=InferenceError),
    infer_scalar(B, int, result=InferenceError),
)
def test_bool_or(x, y):
    return x or y


@mt(infer_standard(B, result=False), infer_standard(None, result=True))
def test_nil_eq(x):
    return x is None


@mt(infer_standard(B, result=True), infer_standard(None, result=False))
def test_nil_ne(x):
    return x is not None


@infer_standard(int, result=0)
def test_bool_ne(x):
    if None:
        return x
    else:
        return 0


@mt(infer_scalar(B, B, result=B), infer_scalar(int, int, result=InferenceError))
def test_and(x, y):
    return x and y


@mt(infer_scalar(int, None, result=int), infer_scalar(int, int, result=int))
def test_and_none(x, y):
    if x > 0 and y is not None:
        return x + y
    else:
        return x


@mt(
    infer_scalar(B, int, int, result=int),
    # Should pass, as switch condition accepts non-bool values
    infer_scalar(int, int, int, result=int),
    infer_scalar(B, int, float, result=InferenceError),
    # Both branches are parsed, so this should fail
    infer_scalar(True, int, float, result=InferenceError),
    infer_scalar(False, int, float, result=InferenceError),
    infer_scalar(True, 1, 2, result=1),
    infer_scalar(False, 1, 2, result=2),
    infer_scalar(B, 1, 2, result=int),
)
def test_switch(c, x, y):
    return switch(c, x, y)


@infer_scalar(int, int, result=int)
def test_switch_switch(x, y):
    def f1(z):
        return z > 0

    def f2(z):
        return z < 0

    f = switch(x > 0, f1, f2)
    return switch(f(y), 1, 2)


# This test raises InferenceError in master branch. I don't know why.
@infer_standard(int, int, result=int)
def test_user_switch_hastype(x, y):
    return user_switch(hastype(x, int), y + 1, y + 2)


@mark_fail(
    AttributeError,
    "``type object 'Thing' has no attribute 'contents' (dataclasses not yet supported)",
)
@infer_standard(B, int, result=int)
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
    infer_scalar(int, Ty(int), result=int),
    infer_scalar(float, Ty(int), result=int),
    infer_scalar(float, Ty(float), result=float),
    # Will currently try to cast float to an object,
    # but object constructor does not take any argument
    infer_scalar(
        float,
        Ty(ANYTHING),
        result=AssertionError("wrong number of arguments, expected 0"),
    ),
    infer_scalar(
        float,
        Ty(Bot),
        result=AssertionError("wrong number of arguments, expected 0"),
    ),
    # Python bool accepts float
    infer_scalar(float, Ty(B), result=bool),
    # Python float accepts bool
    infer_scalar(B, Ty(float), result=float),
)
def test_scalar_cast(x, t):
    return scalar_cast(x, t)


@infer_scalar(int, result=float)
def test_scalar_cast_2(x):
    return scalar_cast(x, float)


@infer_scalar(int, result=float)
def test_scalar_cast_3(x):
    return scalar_cast(x, float)


@infer_scalar(int, int, result=TypeError("Unknown function"))
def test_call_nonfunc(x, y):
    return x(y)


@mt(
    infer_scalar(int, int, int, result=int),
    infer_scalar(float, float, float, result=InferenceError),
)
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


###########################
# Using standard_pipeline #
###########################


@mt(
    infer_standard(int, int, result=int),
)
def test_max_std(x, y):
    if x > y:
        return x
    else:
        return y


@mark_fail(
    AttributeError,
    "`type object 'Point' has no attribute 'x'` (dataclasses not yet supported)",
)
@mt(
    infer_scalar(Point(int, int), result=int),
    infer_scalar(Point(float, float), result=float),
)
def test_class(pt):
    return pt.x + pt.y


@mark_fail(
    AssertionError,
    "`getattr can currently only be used for methods` (dataclasses not yet supported)",
)
@mt(
    infer_scalar(Point(int, int), result=int),
    infer_scalar(Point(float, float), result=float),
)
def test_dataclass_method(pt):
    return pt.abs()


@mark_fail(
    AssertionError,
    "`getattr can currently only be used for methods` (dataclasses not yet supported)",
)
@mt(
    infer_scalar(Point(int, int), result=int),
    infer_scalar(Point(float, float), result=float),
)
def test_dataclass_property(pt):
    return pt.absprop


@mark_fail(
    TypeError,
    "`No __add__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_add(pt1, pt2):
    return pt1 + pt2


@mark_fail(
    TypeError,
    "`No __add__ inference for <class 'myia.testing.common.Point'> + <class 'int'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), result=Point(int, int))
def test_arithmetic_data_add_ct(pt):
    return pt + 10


@mark_fail(
    TypeError,
    "`No __sub__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_sub(pt1, pt2):
    return pt1 - pt2


@mark_fail(
    TypeError,
    "`No __mul__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_mul(pt1, pt2):
    return pt1 * pt2


@mark_fail(
    TypeError,
    "`No __truediv__ method for type <class 'myia.testing.common.Point3D'>` (dataclasses not yet supported)",
)
@infer_standard(
    Point3D(float, float, float),
    Point3D(float, float, float),
    result=Point3D(float, float, float),
)
def test_arithmetic_data_truediv(pt1, pt2):
    return pt1 / pt2


@mark_fail(
    TypeError,
    "`No __floordiv__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_floordiv(pt1, pt2):
    return pt1 // pt2


@mark_fail(
    TypeError,
    "`No __mod__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_mod(pt1, pt2):
    return pt1 % pt2


@mark_fail(
    TypeError,
    "`No __pow__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), Point(int, int), result=Point(int, int))
def test_arithmetic_data_pow(pt1, pt2):
    return pt1 ** pt2


@mark_fail(
    TypeError,
    "`No __pos__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), result=Point(int, int))
def test_arithmetic_data_pos(pt):
    return +pt


@mark_fail(
    TypeError,
    "`No __neg__ method for type <class 'myia.testing.common.Point'>` (dataclasses not yet supported)",
)
@infer_standard(Point(int, int), result=Point(int, int))
def test_arithmetic_data_neg(pt):
    return -pt


@mark_fail(
    AttributeError,
    "`type object 'Point' has no attribute 'x'` (dataclasses not yet supported)",
)
@mt(
    infer_scalar(int, int, int, int, result=Point(int, int)),
    infer_scalar(float, float, float, float, result=InferenceError),
)
def test_dataclass_inst(x1, y1, x2, y2):
    pt1 = Point(x1, y1)
    pt2 = Point(x2, y2)
    return Point(pt1.x + pt2.x, pt1.y + pt2.y)


@pytest.mark.xfail(
    strict=True, reason="Wont' raise (dataclasses not yet supported)"
)
@infer_scalar(int, int, int, result=InferenceError)
def test_dataclass_bad_inst(x, y, z):
    return Point(x, y, z)


@mark_fail(
    AttributeError,
    "`type object 'Point' has no attribute 'z'` (dataclasses not yet supported)",
)
@infer_scalar(Point(int, int), result=InferenceError)
def test_dataclass_wrong_field(pt):
    return pt.z


@mark_fail(
    TypeError, "`Unknown function Thing` (dataclasses not yet supported)"
)
@infer_scalar(Thing(int), result=int)
def test_dataclass_call(thing):
    return thing()


@mt(
    infer_scalar(int, result=0),
    infer_scalar(float, result=0.0),
)
def test_zeros_like(x):
    return zeros_like(x)


@mark_fail(RuntimeError, "No inference for node")
@mt(
    infer_scalar([float], result=[float]),  # list element types are broadened
    infer_scalar((int, float), result=(0, 0.0)),
    infer_scalar((2, 3.0), result=(0, 0.0)),
)
def test_zeros_like_fail(x):
    return zeros_like(x)


@mt(
    infer_scalar(int, result=int),
    infer_scalar(float, result=int),
    infer_scalar((int, int), result=int),
)
def test_unsafe_static_cast(x):
    return unsafe_static_cast(x, int)


@mt(
    infer_scalar(int, int, result=TypeError("Unknown function")),
    infer_scalar(int, (int, int), result=TypeError("Unknown function")),
)
def test_unsafe_static_cast_error(x, y):
    return unsafe_static_cast(x, y)


@infer_scalar(int, result=int)
def test_pass(x):
    if x < 0:
        x = -x
    else:
        pass
    return x


@mt(
    infer_standard(None, float, result=False),
    infer_standard(B, float, result=False),
    infer_standard(float, float, result=MyiaTypeError),
    infer_standard(B, B, result=B),
    infer_standard(None, None, result=True),
    infer_standard(None, NotImplemented, result=False),
    infer_standard(NotImplemented, NotImplemented, result=True),
)
def test_is(x, y):
    return x is y


@mt(
    infer_standard(None, float, result=True),
    infer_standard(B, float, result=True),
    infer_standard(float, float, result=MyiaTypeError),
    infer_standard(B, B, result=B),
    infer_standard(None, None, result=False),
    infer_standard(None, NotImplemented, result=True),
    infer_standard(NotImplemented, NotImplemented, result=False),
)
def test_is_not(x, y):
    return x is not y


@mt(
    infer_standard("idk", result=int),
    infer_standard("hey", result=2),
    infer_standard(String, result=int),
)
def test_string_eq(s):
    x = 2
    if s == "idk":
        x = x + 1
    return x


@mt(
    infer_standard("idk", result=2),
    infer_standard("hey", result=int),
    infer_standard(String, result=int),
)
def test_string_ne(s):
    x = 2
    if s != "idk":
        x = x + 1
    return x


@mt(infer_standard("hey", result="hey"), infer_standard(String, result=String))
def test_string_return(s):
    return s
