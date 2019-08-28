
import operator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest

from myia import abstract
from myia.abstract import (
    ANYTHING,
    CONTEXTLESS,
    Contextless,
    concretize_abstract,
    from_value,
)
from myia.abstract.prim import UniformPrimitiveInferrer
from myia.composite import gadd, zeros_like
from myia.dtype import (
    Array,
    EnvType as Env,
    External,
    Int,
    Nil,
    Number,
    f16,
    f32,
    f64,
    i16,
    i32,
    i64,
    u64,
)
from myia.hypermap import HyperMap, hyper_map
from myia.ir import Graph, MetaGraph, MultitypeGraph
from myia.macros import grad
from myia.operations import user_switch
from myia.pipeline import scalar_pipeline, standard_pipeline
from myia.prim import Primitive, ops as P
from myia.prim.py_implementations import (
    J,
    Jinv,
    array_map,
    array_reduce,
    array_to_scalar,
    bool_and,
    bool_or,
    broadcast_shape,
    casttag,
    dict_setitem,
    distribute,
    dot,
    embed,
    env_getitem,
    env_setitem,
    hastag,
    hastype,
    identity,
    make_record,
    partial as myia_partial,
    record_setitem,
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
)
from myia.utils import InferenceError, MyiaTypeError, newenv

from .common import (
    AA,
    JT,
    TU,
    B,
    Bot,
    D,
    EmptyTuple,
    Ex,
    Pair,
    Point,
    Point3D,
    S,
    Shp,
    Thing,
    Thing_f,
    Thing_ftup,
    Ty,
    U,
    af16_of,
    af32_of,
    af64_of,
    ai32_of,
    ai64_of,
    countdown,
    make_tree,
    mysum,
    to_abstract_test,
)

ai64 = Array[i64]
af64 = Array[f64]


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
                    rval = concretize_abstract(rval)
                    print(rval)
                    return rval

                print('Expected:')
                print(expected_out)

                if _is_exc_type(expected_out):
                    try:
                        out()
                    except expected_out as e:
                        pass
                    else:
                        raise Exception(
                            f'Expected {expected_out}, got: (see stdout).'
                        )
                else:
                    assert out() == expected_out

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
       ([i64], [i64], [[i64]]),
       # ([ai64_of(8, 3)], [ai64_of(4, 3)], [[ai64_of(ANYTHING, 3)]]),
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


@infer(([],))
def test_list_empty():
    return []


@infer(
    (1, D(x=1),),
    (f32, D(x=f32),),
)
def test_dict(x):
    return {'x': x}


@infer(
    (i64, f32, D(x=i64, y=f32)),
)
def test_dict2(x, y):
    return {'x': x, 'y': y}


@infer((i64, i64, f32, D(x=i64, y=f32)))
def test_dict_merge(c, x, y):
    if c == 0:
        return {'x': 1, 'y': 2}
    elif c == 1:
        return {'x': 2, 'y': 4}
    else:
        return {'x': x, 'y': y}


@infer((B, i64, f32, MyiaTypeError))
def test_dict_incompatible(c, x, y):
    if c:
        return {'x': x, 'y': y}
    else:
        return {'x': x, 'yy': y}


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
    ([f64], InferenceError),
    (af64_of(2, 5), i64),
    (i64, InferenceError),
)
def test_array_len(xs):
    return P.array_len(xs)


@infer((i64, f64, i64), (f64, i64, f64))
def test_tuple_getitem(x, y):
    return (x, y)[0]


@infer((i64, f64, f64), (f64, i64, i64))
def test_tuple_getitem_negative(x, y):
    return (x, y)[-1]


@infer((i64, f64, InferenceError))
def test_tuple_outofbound(x, y):
    return (x, y)[2]


@infer(
    ((i64, f64), (f64,)),
    ((f64, i64), (i64,)),
    ((f64, (i64, f64)), ((i64, f64),)),
    ((), ()),
    (f64, InferenceError),
)
def test_tuple_getslice(tup):
    return tup[1:]


@infer(
    ((i64, f64, i64), (f64,)),
    ((f64,), ()),
)
def test_tuple_getslice_2(tup):
    return tup[1:-1]


@infer_std(
    ((i64, i64), (i64,), (i64, i64, i64)),
    ((i64, i64), i64, InferenceError)
)
def test_concat_tuple(x, y):
    return x + y


@infer((i64, f64, InferenceError))
def test_tuple_outofbound_negative(x, y):
    return (x, y)[-3]


@infer((D(x=i64), i64),
       (D(y=f32), InferenceError))
def test_dict_getitem(d):
    return d['x']


@infer((D(x=i64), Ex(ANYTHING, t=str), InferenceError),
       (D(x=i64), 2, InferenceError))
def test_dict_getitem_nonconst(d, i):
    return d[i]


@infer((D(x=i64), f64, D(x=f64)),
       (D(x=i64, y=f32), f64, D(x=f64, y=f32)),
       (D(z=i64), f64, InferenceError))
def test_dict_setitem(d, x):
    return dict_setitem(d, 'x', x)


@infer(
    ((i64, i64), 1, f64, (i64, f64)),
    ((i64, i64, f64), 1, f64, (i64, f64, f64)),
    ((i64,), 1, f64, InferenceError),
    ((i64,), 0.0, f64, InferenceError),
    ((i64,), i64, f64, InferenceError),
)
def test_tuple_setitem(xs, idx, x):
    return tuple_setitem(xs, idx, x)


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
    (i64, i64, i64),
    (InferenceError,),
    (i64, i64, i64, InferenceError),
)
def test_default_arg(x, y=3):
    return x + y


@infer((i64, i64, i64))
def test_default_closure(x, y):
    def clos(z=y + y, q=x + x):
        return x + z

    return clos(y)


@infer_std((1,),)
def test_closure_manager_bug():
    rval = 0
    for z in (1, 2, 3, 4):
        if z == 1:
            rval = z
    return rval


@infer(
    (0,),
    (i64, i64, i64),
    (i64, i64, i64, i64, i64, i64, i64),
)
def test_varargs(*args):
    rval = 0
    for arg in args:
        rval = rval + arg
    return rval


@infer(
    (i64, i64, i64),
)
def test_keywords(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, beatrice=y) + fn(beatrice=3, albert=7)


@infer(
    (i64, i64, i64),
)
def test_keywords_expand(x, y):
    def fn(z, albert, beatrice):
        return albert - beatrice + z

    return fn(4, **{'albert': x, 'beatrice': y})


@infer(
    (i64, i64, InferenceError),
)
def test_keywords_bad(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, charles=y)


@infer(
    (i64, i64, i64),
)
def test_keywords_different_order(x, y):
    def fn1(x, albert, beatrice):
        return albert * (x - beatrice)

    def fn2(y, beatrice, albert):
        return y * (albert - beatrice)

    fn = fn1 if x < 0 else fn2

    return fn(5, albert=x, beatrice=y)


@infer((i64, i64, i64),)
def test_keywords_defaults(x, y):
    def fn(charles, *, albert=1, beatrice=10):
        return albert - beatrice + charles

    return fn(x, beatrice=y)


@infer((i64, i64, i64),)
def test_keywords_shadow(x, y):
    # It used to be that the beatrice arg would be renamed barbara
    # because of the assignment.
    def fn(albert, beatrice):
        barbara = beatrice
        return albert - barbara

    return fn(albert=x, beatrice=y)


@infer((i64, i64, InferenceError),)
def test_redundant_kw(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, **{'albert': y, 'beatrice': y})


@infer((i64, i64),)
def test_defaults_recursive(x):
    def fact(n=x):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact()


@infer((i64, i64, (i64, i64, i64)),)
def test_kwarg(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(x, beatrice=y), proxy(x, y), proxy(beatrice=x, albert=y)


@infer((i64, i64, InferenceError),)
def test_kwarg_bad(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(albert=x, beatrice=y, charles=x + y)


@infer(
    (i64, i64, InferenceError),
)
def test_keywords_bad_3(x, y):
    return scalar_add(x=x, y=y)


@infer(
    ((i64, i64, i64), i64),
    ((i64, i64, f64), InferenceError),
    ((i64, i64, i64, i64), InferenceError),
    ((i64, i64), InferenceError),
    (i64, InferenceError),
)
def test_apply(args):
    def _f(x, y, z):
        return x + y + z

    return _f(*args)


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
    (i64, True),
    (f64, False),
)
def test_hastype_simple(x):
    return hastype(x, i64)


@infer(
    (i64, i64, InferenceError),
    (i64, Ty(i64), True),
    (f64, Ty(i64), False),
    ((i64, i64), Ty(tuple), True),
    ((i64, i64), Ty(Tuple), True),
    ((i64, i64), Ty(Tuple[Number, Number]), True),
    ((i64, i64), Ty(Tuple[i64, i64]), True),
    ((i64, i64), Ty(Tuple[float, float]), False),
    ((i64, i64), Ty(ANYTHING), InferenceError),
    ([i64], Ty(List), True),
    (None, Ty(Nil), True),
    (U(i32, i64), Ty(i64), B),
    (i32, Ty(U(i16, i32)), True),
    (U(i32, i64), Ty(U(i16, i32)), B),
)
def test_hastype(x, y):
    return hastype(x, y)


@infer(
    (i64, Ty(to_abstract_test(i64))),
    (f64, Ty(to_abstract_test(f64))),
)
def test_typeof(x):
    return typeof(x)


Tf4 = Tuple[f64, f64, f64, f64]


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
    (Thing_ftup, (f64, f64)),
    (Thing_f, 0),
    (5, 5),
    (Point3D(5, 7, 9), 0),
    (U(ai64_of(2, 5), [f32]), f64),
    (U([i64], [f32]), 1.0),
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


@pytest.mark.xfail(
    reason="user_switch can't process conditions that partly depend on type"
)
@infer_std(
    (U(i64, (i64, i64)), i64),
)
def test_union_2(x):
    def condition(x):
        return hastype(x, i64) and x > 0

    if condition(x):
        return x
    else:
        return -1


@infer_std(
    (U(i64, None), i64),
    (None, 0),
)
def test_union_nil(x):
    if x is None:
        return 0
    else:
        return x


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
    (TU(_1=i64, _2=f64, _37=(i64, i64)), i64),
    (TU(_2=f64, _3=(i64, i64)), InferenceError),
    (TU(_1=i64, _2=f64, _77=(i64, i64)), InferenceError),
)
def test_hastag_casttag(x):
    if hastag(x, 1):
        return casttag(x, 1)
    elif hastag(x, 2):
        return 1234
    else:
        return casttag(x, 37)[0]


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


@infer(
    (U(C1(2), C2(5)), i64, i64)
)
def test_getattr_union(c, x):
    return c.f(x)


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
     Ex('surprise'),
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
    (B, B, B),
    (i64, i64, InferenceError),
)
def test_and(x, y):
    return x and y


@infer(
    (i64, None, i64),
    (i64, i64, i64),
)
def test_and_none(x, y):
    if x > 0 and y is not None:
        return x + y
    else:
        return x


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
    (i64, i64, InferenceError),
)
def test_user_switch_hastype(x, y):
    return user_switch(hastype(x, i64), y + 1, y + 2)


@infer((B, i64, i64))
def test_closure_in_data(c, x):
    def f(x):
        return x * x

    def g(x):
        return x + x

    a = Thing((1, [f]))
    b = Thing((2, [g]))
    _, h = switch(c, a, b).contents
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
    return scalar_to_array(x, AA)


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


class _BadMG(MetaGraph):
    def generate_graph(self, sig):
        return None


_bmg = _BadMG('badmg')


@infer((i64, InferenceError))
def test_bad_metagraph(x):
    return _bmg(x)


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
    if hastype(x, Tuple):
        return x[0]
    else:
        return x


@infer(
    (i64, i64),
    (f64, f64),
)
def test_add1_hastype_interference(x):
    return x + _interference_helper(1)


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


@infer(
    (Point(i64, i64), i64),
    (Point(f64, f64), f64),
)
def test_dataclass_property(pt):
    return pt.absprop


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


@infer((Thing(i64), f64, Thing(f64)))
def test_record_setitem(thing, x):
    return record_setitem(thing, 'contents', x)


@infer((Point(i64, i64), i64, Point(i64, i64)),
       (Point(i64, i64), f64, InferenceError))
def test_record_setitem_2(pt, x):
    return record_setitem(pt, 'x', x)


@infer((Thing(i64), f64, InferenceError))
def test_record_setitem_wrong_field(thing, x):
    return record_setitem(thing, 'shfifty_five', x)


hyper_map_notuple = HyperMap(
    nonleaf=(abstract.AbstractArray,
             abstract.AbstractUnion,
             abstract.AbstractClassBase)
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
    (U(i64, (i64, i64)), U(i64, (i64, i64)), U(i64, (i64, i64))),
    (U(i64, (i64, i64)), U(i64, (i64, i64, i64)), InferenceError),
    ({"x": i64, "y": i64}, {"x": i64, "y": i64}, {"x": i64, "y": i64}),
    ({"x": i64, "y": i64}, {"x": i64, "y": i64, "z": i64}, InferenceError),
    ({"x": i64, "y": i64}, {"y": i64, "z": i64}, InferenceError),

    # Generic broadcasting tests
    ([f64], f64, [f64]),
    ([[f64]], [[f64]], [[f64]]),
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
    (ai64_of(2, 1), ai64_of(1, 5), InferenceError),
    (Env, Env, Env),
    ({"x": i64, "y": i64}, {"x": i64, "y": i64}, {"x": i64, "y": i64}),
)
def test_gadd(x, y):
    return gadd(x, y)


@infer(
    (i64, 0),
    (f64, 0.0),
    ([f64], [f64]),  # list element types are broadened
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


@infer((i32, i32))
def test_raise(x):
    if x >= 0:
        return x ** 0.5
    else:
        raise Exception("sqrt of negative number")


@infer((i32, Bot))
def test_raise_unconditional(x):
    raise Exception("I don't like your face")


@infer((i32, i32))
def test_raise_multiple(x):
    if x < 0:
        raise Exception("You are too ugly")
    elif x > 0:
        raise Exception("You are too beautiful")
    else:
        return x


@infer((i32, Bot), (f64, f64), (U(i32, i64), i64))
def test_raise_hastype(x):
    if hastype(x, i32):
        raise Exception("What a terrible type")
    else:
        return x


@infer((i32, i32))
def test_raise_loop(x):
    while x < 100:
        x = x * x
        if x > 150:
            raise Exception("oh no")
    return x


@infer((i32, i32))
def test_raise_rec(x):
    def f(x):
        if x == 0:
            return 1
        elif x >= 10:
            raise Exception("too big")
        else:
            return x * f(x - 1)
    return f(x)


@infer((i64, i32, i64, U(i32, i64)))
def test_tagged(x, y, z):
    if x > 0:
        return tagged(y)
    else:
        return tagged(z)


@infer((i64, i16, i32, i64, U(i16, i32, i64)))
def test_tagged_more(c, x, y, z):
    if c == 0:
        return tagged(x)
    elif c > 0:
        return tagged(y)
    else:
        return tagged(z)


@infer((i64, InferenceError))
def test_tagged_too_many_arguments(x):
    return tagged(x, 1, 2)


pair_t1 = from_value(Pair(Pair(1, 2), Pair(2, 3)))
pair_t1_u = pair_t1.attributes['left']


@infer((i64, pair_t1_u))
def test_tagged_adt(depth):
    return make_tree(depth, 1)


pair_t2 = from_value(Pair(1, Pair(2, Pair(3, None))))
pair_t2_u = pair_t2.attributes['right']


@infer((i64, pair_t2_u))
def test_tagged_adt_2(depth):
    return countdown(depth)


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


@infer_std(
    (i64, i64),
    (f32, f32),
    (f64, f64),
)
def test_grad_cast(x):
    def f(x):
        return scalar_cast(x, f16)

    return grad(f)(x)


@infer_std(
    (af16_of(2, 5), af16_of(2, 5), af16_of(2, 5)),
)
def test_grad_reduce(xs, ys):
    def f(xs, ys):
        return array_reduce(scalar_add, xs * ys, ())

    return grad(f)(xs, ys)


@infer_std(
    (None, f64, False),
    (B, f64, False),
    (f64, f64, MyiaTypeError),
    (B, B, B),
    (None, None, True),
)
def test_is(x, y):
    return x is y


@infer_std(
    (None, f64, True),
    (B, f64, True),
    (f64, f64, MyiaTypeError),
    (B, B, B),
    (None, None, False),
)
def test_is_not(x, y):
    return x is not y


@infer(
    (af32_of(1, 3, 4, 5), af32_of(3, 1, 3, 3), Shp(2, 3), Shp(3, 2),
     Shp(3, 4), u64, af32_of(1, 3, 2, 1)),
    (af32_of(2, 3, 4, 5), af32_of(3, 1, 3, 3), Shp(2, 3), Shp(3, 2),
     Shp(3, 4), u64, af32_of(2, 3, 2, 1)),
    (af32_of(2, 3, 4, 5), af32_of(3, 1, 3, 3), Shp(2, 3, 4), Shp(3, 2),
     Shp(3, 4), u64, InferenceError),
    (af32_of(2, 3, 4, 5), af32_of(3, 1, 3, 3), Shp(2, 3), Shp(3, 2, 4),
     Shp(3, 4), u64, InferenceError),
    (af32_of(2, 3, 4, 5), af32_of(3, 1, 3, 3), Shp(2, 3), Shp(3, 2),
     Shp(3, 4, 2), u64, InferenceError),
)
def test_conv2d(i, w, s, p, d, g):
    return P.conv2d(i, w, s, p, d, g)


@infer(
    (Shp(1, 3, 4, 5), af32_of(3, 1, 3, 3), af32_of(1, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(1, 3, 4, 5)),
    (Shp(2, 3, 4, 5), af32_of(3, 1, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(2, 3, 4, 5)),
    (Shp(2, 6, 4, 5), af32_of(3, 2, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(2, 6, 4, 5)),
    (Shp(2, 1, 4, 5), af32_of(3, 1, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(1, u64), af32_of(2, 1, 4, 5)),
)
def test_conv2d_input_grad(i_s, w, g_o, s, p, d, g):
    return P.conv2d_input_grad(i_s, w, g_o, s, p, d, g)


@infer(
    (af32_of(1, 3, 4, 5), Shp(3, 1, 3, 3), af32_of(1, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(3, 1, 3, 3)),
    (af32_of(2, 3, 4, 5), Shp(3, 1, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(3, 1, 3, 3)),
    (af32_of(2, 6, 4, 5), Shp(3, 2, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(3, u64), af32_of(3, 2, 3, 3)),
    (af32_of(2, 1, 4, 5), Shp(3, 1, 3, 3), af32_of(2, 3, 2, 1), Shp(2, 3),
     Shp(3, 2), Shp(3, 4), S(1, u64), af32_of(3, 1, 3, 3)),
)
def test_conv2d_weight_grad(i, w_s, g_o, s, p, d, g):
    return P.conv2d_weight_grad(i, w_s, g_o, s, p, d, g)
