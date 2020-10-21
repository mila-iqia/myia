import numpy as np
from pytest import mark

from myia.hypermap import hyper_map
from myia.operations import (
    array_map,
    array_reduce,
    partial,
    scalar_add,
    scalar_sub,
    scalar_uadd,
    scalar_usub,
    switch,
    tagged,
)
from myia.pipeline import scalar_debug_pipeline, standard_debug_pipeline, steps
from myia.testing.common import Point, U, f64, i64, mysum
from myia.testing.multitest import mt, run

mono_pipeline = scalar_debug_pipeline.select(
    "resources",
    "parse",
    "infer",
    "specialize",
    "simplify_types",
    {"opt2": steps.step_opt2_no_main},
    "llift",
    "validate",
    "export",
    "wrap",
)


mono_pipeline_std = standard_debug_pipeline.select(
    "resources",
    "parse",
    "infer",
    "specialize",
    "simplify_types",
    "opt",
    "opt2",
    "llift",
    "validate",
    "export",
    "wrap",
)


mono_scalar = run.configure(pipeline=mono_pipeline, backend=False)
mono_standard = run.configure(pipeline=mono_pipeline_std, backend=False)


int1 = 13
int2 = 21

int1_np64 = np.int64(17)
int2_np64 = np.int64(29)

int1_np32 = np.int32(37)
int2_np32 = np.int32(41)

fp1 = 2.7
fp2 = 6.91

fp1_np64 = np.float64(3.3)
fp2_np64 = np.float64(7.23)

fp1_np32 = np.float32(3.9)
fp2_np32 = np.float32(9.29)

pt1 = Point(10, 20)
pt2 = Point(100, 200)


@mt(
    mono_scalar(int1, int2_np64),
    mono_scalar(int1_np64, int2_np64),
    mono_scalar(fp1, fp2_np64),
    mono_scalar(fp1_np64, fp2_np64),
    mono_scalar(fp1_np32, fp2_np32),
    mono_scalar(int1_np32, int2_np32),
)
def test_prim_arithmetic_np_same_precision(x, y):
    def test_prim_mul_np(x, y):
        return x * y

    def test_prim_add_np(x, y):
        return x + y

    def test_prim_sub_np(x, y):
        return x - y

    def test_prim_div_np(x, y):
        return x / y

    _a = test_prim_mul_np(x, y)
    _b = test_prim_add_np(x, y)
    _c = test_prim_sub_np(x, y)
    _d = test_prim_div_np(x, y)

    return _a, _b, _c, _d


@mt(mono_scalar(int1, int2), mono_scalar(fp1, fp2))
def test_prim_mul(x, y):
    return x * y


@mt(mono_scalar(int1, int2), mono_scalar(fp1, int1))
def test_polymorphic(x, y):
    def helper(a, b):
        return a * a + b * b

    return helper(x, x + x), helper(y, y + y)


@mt(mono_scalar(int1, int2), mono_scalar(fp1, int1))
def test_polymorphic_closure(x, y):
    def construct(z):
        def inner(w):
            return z + w

        return inner

    return construct(x + x)(x), construct(y + y)(y)


@mt(mono_scalar(True, int1, int2), mono_scalar(True, fp1, int1))
def test_switch_fn(c, x, y):
    def dee(y):
        return y * y

    def doo(y):
        return y + y

    if c:
        f = dee
    else:
        f = doo

    return f(x), f(y)


@mt(mono_scalar(int1, int2), mono_scalar(int1, fp1))
def test_while(n, x):
    rval = x
    while n > 0:
        n = n - 1
        rval = rval - x
    return rval


@mono_scalar(int1, fp1)
def test_isinstance(x, y):
    def helper(x):
        if isinstance(x, int):
            return x
        elif isinstance(x, float):
            return x
        else:
            return (x,)

    return helper(x), helper(y), helper(())


@mono_scalar(int1, int2)
def test_struct(x, y):
    return Point(x, y)


@mono_scalar(int1, int2)
def test_struct2(x, y):
    p = Point(x, y)
    return p.x + p.y


@mono_scalar(np.array([fp1, fp2]))
def test_array_map(xs):
    def square(x):
        return x * x

    return array_map(square, xs)


@mono_scalar(np.array([fp1, fp2]), np.array([int1, int2]))
def test_array_map_polymorphic(xs, ys):
    def square(x):
        return x * x

    return array_map(square, xs), array_map(square, ys)


@mono_scalar(np.array([fp1, fp2]), np.array([int1, int2]))
def test_array_map_polymorphic_indirect(xs, ys):
    def square(x):
        return x * x

    def helper(fn):
        return array_map(fn, xs), array_map(fn, ys)

    return helper(square)


@mono_scalar(np.array([fp1, fp2]), np.array([int1, int2]))
def test_array_reduce_polymorphic_indirect(xs, ys):
    def helper(fn):
        return array_reduce(fn, xs, ()), array_reduce(fn, ys, ())

    return helper(scalar_add)


@mono_scalar(int1, np.array([fp1, fp2]))
def test_array_map_partial(c, xs):
    def square(x):
        return x * x

    def identity(x):
        return x

    if c < 0:
        fn = partial(array_map, square)
    else:
        fn = identity
    return fn(xs)


@mono_scalar([fp1, fp2])
def test_list_len(xs):
    return len(xs)


@mono_scalar([fp1, fp2])
def test_hyper_map(xs):
    def square(x):
        return x * x

    return hyper_map(square, xs)


@mono_scalar([fp1, fp2], [int1, int2])
def test_hyper_map_polymorphic(xs, ys):
    def square(x):
        return x * x

    return hyper_map(square, xs), hyper_map(square, ys)


@mark.xfail(reason="Cannot specialize f.")
@mono_scalar(True, [fp1, fp2], [int1, int2])
def test_hyper_map_polymorphic_2(c, xs, ys):
    def square(x):
        return x * x

    def double(x):
        return x + x

    def picker(c):
        if c:
            return square
        else:
            return double

    f = picker(c)
    return hyper_map(f, xs), hyper_map(f, ys)


@mono_scalar(0, 8)
def test_mutual_recursion_closure(start, n):
    def _even(n):
        if n == start:
            return True
        else:
            return _odd(n - 1)

    def _odd(n):
        if n == start:
            return False
        else:
            return _even(n - 1)

    return _even(n)


@mono_scalar(int1, int2)
def test_unused_parameter(x, y):
    return x * x


@mono_scalar(int1)
def test_unused_function_parameter(x):
    # The type of square will be AbstractError(DEAD), but that's not really
    # an issue because it is indeed not used, and we can simply replace the
    # reference by a dummy.
    def square(y):
        return y * y

    def helper(f, a):
        return a * a

    return helper(square, x)


@mono_scalar(int1)
def test_indirect_primitive(x):
    def add2():
        return scalar_add

    return add2()(x, x)


@mono_scalar(int1)
def test_indirect_graph(x):
    def f(x):
        return x * x

    def f2():
        return f

    return f2()(x)


@mono_scalar(True, int1, int2)
def test_poly_with_constants(c, x, y):
    def f1(x, y):
        return x + y

    def f2(x, y):
        return x * y

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, y), choose(not c)(x, y)


@mono_scalar(True, int1, int2)
def test_poly_with_constants2(c, x, y):
    def f1(x, y):
        return x + y

    def f2(x, y):
        return x * y

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, 2), choose(not c)(y, 3)


@mt(mono_scalar(int1, int2), mono_scalar(fp1, fp2))
def test_method(x, y):
    return x.__add__(y)


@mono_scalar(int1, fp1)
def test_method_polymorphic(x, y):
    return x.__add__(x), y.__add__(y)


@mono_scalar(int1, fp1)
def test_partial_polymorphic(x, y):
    def f(a, b):
        return a + b

    return partial(f, x)(x), partial(f, y)(y)


@mt(mono_scalar(True, int1), mono_scalar(False, int1))
def test_switch(c, x):
    return switch(c, scalar_usub, scalar_uadd)(x)


@mt(mono_scalar(True, int1, int2), mono_scalar(False, int1, int2))
def test_switch2(c, x, y):
    fn = switch(c, partial(scalar_sub, x), partial(scalar_add, x))
    return fn(y)


@mono_scalar(int1, int2, int2)
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


@mono_scalar(int1, int2)
def test_closure_stays_in_scope(x, y):
    # The inferrer knows that h(x + y) is the graph for g, but
    # it shouldn't try to replace the expression with that graph,
    # because it points to a fv in f.
    def f(z):
        def g():
            return z

        return g

    def h(z):
        a = z * z
        return f(a)

    return h(x + y)()


@mono_scalar(int1)
def test_return_closure(x):
    # The specializer should be careful not to replace `f(z - 1)[0]`
    # by a reference to `g`, because `g` is closed over `z` whereas
    # `f(z - 1)[0]` refers to a version of `g` closed on `z - 1`.
    def f(z):
        def g():
            return z

        def h():
            return f(z - 1)[0]()

        return (g, h)

    return f(x)[1]()


@mono_scalar(int1, int2)
def test_partial_outside_scope(x, y):
    # The inferrer knows that g(x) is a partial of f, but it can't
    # build it inside the main function.
    def f(x, y):
        return x * y

    def g(x):
        z = x * x
        return partial(f, z)

    return g(x)(y)


@mt(
    mono_standard(int1),
    mono_standard([int1, int2]),
    mono_standard((int1, int2), result=TypeError),
    abstract=(U(i64, [i64]),),
)
def test_union(x):
    if isinstance(x, int):
        return x
    else:
        return x[0]


@mt(
    mono_standard(int1, int2),
    mono_standard([int1, int2], int1),
    mono_standard((int1, int2), int1, result=TypeError),
    abstract=(U(i64, f64, [i64]), i64),
)
def test_union_nested(x, y):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return y
    else:
        return x[0]


@mt(
    mono_standard(int1, int2),
    mono_standard([int1, int2], int1),
    mono_standard((int1, int2), int1, result=TypeError),
    abstract=(U(i64, f64, [i64]), i64),
)
def test_union_nested_2(x, y):
    if isinstance(x, (int, float)):
        if isinstance(x, int):
            return x
        else:
            return y
    else:
        return x[0]


@mt(
    mono_scalar(1, fp1, pt1, (int1, int2)),
    mono_scalar(0, fp1, pt1, (int1, int2)),
    mono_scalar(-1, fp1, pt1, (int1, int2)),
)
def test_tagged(c, x, y, z):
    if c > 0:
        return tagged(x)
    elif c == 0:
        return tagged(y)
    else:
        return tagged(z)


@mono_standard((int1, int2), (fp1, fp2))
def test_tuple_surgery(xs, ys):
    return xs[::-1]


@mt(
    mono_scalar(int1),
    mono_scalar(fp1),
    mono_scalar(int1, int2),
    mono_scalar(int1, int2, int1),
)
def test_default_arg(x, y=3, z=6):
    return x + y + z


@mono_scalar(int1, int2)
def test_default_closure(x, y):
    def clos(z=y + y):
        if x > z:
            return x - z
        else:
            return 0

    return clos() + clos(x)


@mono_scalar(1, 2, 3, 4, 5)
def test_varargs(x, *args):
    return args[1]


def _v(x, *args):
    return x + args[-1]


@mono_scalar(int1, int2)
def test_varargs_2(x, y):
    argos = (1, 2, 3, 4, 5)
    return _v(x, 1, 2, 3) + _v(y, 5) + _v(*argos, 6, *(4, 5))


@mono_scalar(int1, int2)
def test_keywords(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, beatrice=y) + fn(beatrice=3, albert=7)


@mono_scalar(int1, int2)
def test_keywords_2(x, y):
    def fn1(x, albert, beatrice):
        return albert * (x - beatrice)

    def fn2(y, albert, beatrice):
        return y * (albert - beatrice)

    fn = fn1 if x < 0 else fn2

    return fn(5, albert=x, beatrice=y) + fn(9, albert=3, beatrice=7)


@mono_scalar(int1, int2)
def test_keywords_defaults(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    return fn(beatrice=x + y)


@mono_scalar(int1, int2)
def test_kwarg(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(x, beatrice=y), proxy(x, y), proxy(beatrice=x, albert=y)


@mono_standard()
def test_reference_bug():
    orig = (4, 1, 6)
    shp = ()
    for z in orig:
        if z != 1:
            shp = shp + (z,)
    return shp


@mt(
    mono_standard([11, 22, 33], [44, 55, 66]),
    mono_standard((11, 22, 33), (44, 55, 66)),
    validate=False,
)
def test_zip_enumerate(xs, ys):
    rval = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        rval = rval + i + x + y
    return rval


def list_reduce(fn, lst, dftl):
    res = dftl
    for elem in lst:
        res = fn(res, elem)
    return res


@mono_standard([int1, int2])
def test_list_reduce(xs):
    def add(x, y):
        return x + y

    return list_reduce(add, xs, 4)
