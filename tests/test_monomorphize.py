
import numpy as np
from pytest import mark

from myia.abstract import from_value
from myia.debug.label import short_labeler as lbl
from myia.debug.traceback import print_inference_error
from myia.dtype import Number
from myia.hypermap import hyper_map
from myia.pipeline import scalar_debug_pipeline, standard_debug_pipeline
from myia.prim.py_implementations import (
    array_map,
    array_reduce,
    hastype,
    partial,
    scalar_add,
    scalar_mul,
    scalar_sub,
    scalar_uadd,
    scalar_usub,
    switch,
    tagged,
)
from myia.utils import InferenceError, overload
from myia.validate import ValidationError

from .common import (
    Point,
    U,
    countdown,
    f64,
    i64,
    make_tree,
    mysum,
    reducetree,
    sumtree,
    to_abstract_test,
)

specialize_pipeline = scalar_debug_pipeline \
    .select('parse', 'infer', 'specialize', 'simplify_types',
            'opt2', 'validate', 'export', 'wrap') \
    .configure({
        'opt2.phases.main': [],
    })


specialize_pipeline_std = standard_debug_pipeline \
    .select('parse', 'infer', 'specialize', 'simplify_types',
            'opt', 'opt2', 'validate', 'export', 'wrap')


@overload
def _eq(t1: tuple, t2):
    return (isinstance(t2, tuple)
            and all(_eq(x1, x2) for x1, x2 in zip(t1, t2)))


@overload  # noqa: F811
def _eq(a1: np.ndarray, a2):
    return (a1 == a2).all()


@overload  # noqa: F811
def _eq(x: object, y):
    return x == y


def specializer_decorator(pipeline):
    def specialize(*arglists, abstract=None):

        def decorate(fn):
            def run_test(args):
                if isinstance(args, Exception):
                    exc = type(args)
                    args = args.args
                else:
                    exc = None
                pip = pipeline.make()
                if abstract is None:
                    argspec = tuple(from_value(arg, broaden=True)
                                    for arg in args)
                else:
                    argspec = tuple(to_abstract_test(a) for a in abstract)

                if exc is not None:
                    try:
                        mfn = pip(input=fn, argspec=argspec)
                        mfn['output'](*args)
                    except exc:
                        pass
                    return

                result_py = fn(*args)

                try:
                    res = pip(input=fn, argspec=argspec)
                except InferenceError as ierr:
                    print_inference_error(ierr)
                    raise ierr
                except ValidationError as verr:
                    print('Collected the following errors:')
                    for err in verr.errors:
                        n = err.node
                        nlbl = lbl.label(n)
                        tname = type(n).__name__
                        print(f'   {nlbl} ({tname}) :: {n.abstract}')
                        print(f'      {err.args[0]}')
                    raise verr

                result_final = res['output'](*args)
                assert _eq(result_py, result_final)

            m = mark.parametrize('args', arglists)(run_test)
            m.__orig__ = fn
            return m

        return decorate
    return specialize


specialize = specializer_decorator(specialize_pipeline)
specialize_std = specializer_decorator(specialize_pipeline_std)
specialize_no_validate = specializer_decorator(
    specialize_pipeline.configure(validate=False)
)


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


@specialize((int1, int2_np64), (int1_np64, int2_np64),
            (fp1, fp2_np64), (fp1_np64, fp2_np64),
            (fp1_np32, fp2_np32), (int1_np32, int2_np32))
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


@specialize((int1, int2),
            (fp1, fp2))
def test_prim_mul(x, y):
    return x * y


@specialize((int1, int2),
            (fp1, int1))
def test_polymorphic(x, y):
    def helper(a, b):
        return a * a + b * b
    return helper(x, x + x), helper(y, y + y)


@specialize((int1, int2),
            (fp1, int1))
def test_polymorphic_closure(x, y):
    def construct(z):
        def inner(w):
            return z + w
        return inner
    return construct(x + x)(x), construct(y + y)(y)


@specialize((True, int1, int2),
            # (True, fp1, int1)  # TODO: mark this one as xfail
            )
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


@specialize((int1, int2), (int1, fp1))
def test_while(n, x):
    rval = x
    while n > 0:
        n = n - 1
        rval = rval - x
    return rval


@specialize((int1,), (fp1,))
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


@specialize((int1, fp1))
def test_hastype(x, y):
    def helper(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return x
        else:
            return (x,)

    return helper(x), helper(y), helper(())


@specialize((int1, int2))
def test_struct(x, y):
    return Point(x, y)


@specialize((int1, int2))
def test_struct2(x, y):
    p = Point(x, y)
    return p.x + p.y


@specialize((np.array([fp1, fp2]),))
def test_array_map(xs):
    def square(x):
        return x * x

    return array_map(square, xs)


@specialize((np.array([fp1, fp2]),
             np.array([int1, int2])))
def test_array_map_polymorphic(xs, ys):
    def square(x):
        return x * x

    return array_map(square, xs), array_map(square, ys)


@specialize((np.array([fp1, fp2]),
             np.array([int1, int2])))
def test_array_map_polymorphic_indirect(xs, ys):
    def square(x):
        return x * x

    def helper(fn):
        return array_map(fn, xs), array_map(fn, ys)

    return helper(square)


@specialize((np.array([fp1, fp2]),
             np.array([int1, int2])))
def test_array_reduce_polymorphic_indirect(xs, ys):
    def helper(fn):
        return array_reduce(fn, xs, ()), array_reduce(fn, ys, ())

    return helper(scalar_add)


@specialize((int1, np.array([fp1, fp2]),))
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


@specialize(([fp1, fp2],))
def test_list_len(xs):
    return len(xs)


@specialize(([fp1, fp2],))
def test_hyper_map(xs):
    def square(x):
        return x * x

    return hyper_map(square, xs)


@specialize(([fp1, fp2], [int1, int2]))
def test_hyper_map_polymorphic(xs, ys):
    def square(x):
        return x * x

    return hyper_map(square, xs), hyper_map(square, ys)


@mark.xfail(reason="Cannot specialize f.")
@specialize((True, [fp1, fp2], [int1, int2]))
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


@specialize((int1, int2))
def test_unused_parameter(x, y):
    return x * x


@specialize((int1,))
def test_unused_function_parameter(x):
    # The type of square will be AbstractError(DEAD), but that's not really
    # an issue because it is indeed not used, and we can simply replace the
    # reference by a dummy.
    def square(y):
        return y * y

    def helper(f, a):
        return a * a
    return helper(square, x)


@specialize((int1,))
def test_indirect_primitive(x):
    def add2():
        return scalar_add

    return add2()(x, x)


@specialize((int1,))
def test_indirect_graph(x):
    def f(x):
        return x * x

    def f2():
        return f

    return f2()(x)


@specialize((True, int1, int2))
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


@specialize((True, int1, int2))
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


@specialize((int1, int2), (fp1, fp2))
def test_method(x, y):
    return x.__add__(y)


@specialize((int1, fp1))
def test_method_polymorphic(x, y):
    return x.__add__(x), y.__add__(y)


@specialize((int1, fp1))
def test_partial_polymorphic(x, y):
    def f(a, b):
        return a + b
    return partial(f, x)(x), partial(f, y)(y)


@specialize((True, int1), (False, int1))
def test_switch(c, x):
    return switch(c, scalar_usub, scalar_uadd)(x)


@specialize((True, int1, int2), (False, int1, int2))
def test_switch2(c, x, y):
    fn = switch(
        c,
        partial(scalar_sub, x),
        partial(scalar_add, x)
    )
    return fn(y)


@specialize((int1, int2, int2))
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


@specialize((int1, int2))
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


@specialize((int1,))
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


@specialize((int1, int2))
def test_partial_outside_scope(x, y):
    # The inferrer knows that g(x) is a partial of f, but it can't
    # build it inside the main function.
    def f(x, y):
        return x * y

    def g(x):
        z = x * x
        return partial(f, z)

    return g(x)(y)


@specialize(
    (int1,),
    ([int1, int2],),
    TypeError((int1, int2),),
    abstract=(U(i64, [i64]),)
)
def test_union(x):
    if hastype(x, i64):
        return x
    else:
        return x[0]


@specialize(
    (int1, int2),
    ([int1, int2], int1),
    TypeError((int1, int2), int1),
    abstract=(U(i64, f64, [i64]), i64)
)
def test_union_nested(x, y):
    if hastype(x, i64):
        return x
    elif hastype(x, f64):
        return y
    else:
        return x[0]


@specialize(
    (int1, int2),
    ([int1, int2], int1),
    TypeError((int1, int2), int1),
    abstract=(U(i64, f64, [i64]), i64)
)
def test_union_nested_2(x, y):
    if hastype(x, Number):
        if hastype(x, i64):
            return x
        else:
            return y
    else:
        return x[0]


@specialize(
    (1, fp1, pt1, (int1, int2)),
    (0, fp1, pt1, (int1, int2)),
    (-1, fp1, pt1, (int1, int2)),
)
def test_tagged(c, x, y, z):
    if c > 0:
        return tagged(x)
    elif c == 0:
        return tagged(y)
    else:
        return tagged(z)


@specialize(
    (int1, int2),
    (pt1, int1),
    (pt1, pt2),
)
def test_hyper_map_scalar_add(x, y):
    return hyper_map(scalar_add, x, y)


@specialize((int1,))
def test_hyper_map_ct(x):
    return hyper_map(scalar_add, x, 1)


@specialize(
    (make_tree(3, 1),),
    (countdown(10),)
)
def test_sumtree(t):
    return sumtree(t)


@specialize(
    (make_tree(3, 1),),
    (countdown(10),)
)
def test_reducetree(t):
    return reducetree(scalar_mul, t, 1)


@specialize_no_validate(
    (make_tree(3, 1),),
)
def test_hypermap_tree(t):
    return hyper_map(scalar_add, t, t)


@specialize(((int1, int2), (fp1, fp2)),)
def test_tuple_surgery(xs, ys):
    return xs[::-1]


@specialize(
    (int1,),
    (fp1,),
    (int1, int2),
    (int1, int2, int1),
)
def test_default_arg(x, y=3, z=6):
    return x + y + z


@specialize(
    (int1, int2),
)
def test_default_closure(x, y):
    def clos(z=y + y):
        if x > z:
            return x - z
        else:
            return 0
    return clos() + clos(x)


@specialize(
    (1, 2, 3, 4, 5),
)
def test_varargs(x, *args):
    return args[1]


def _v(x, *args):
    return x + args[-1]


@specialize((int1, int2))
def test_varargs_2(x, y):
    argos = (1, 2, 3, 4, 5)
    return _v(x, 1, 2, 3) + _v(y, 5) + _v(*argos, 6, *(4, 5))


@specialize((int1, int2),)
def test_keywords(x, y):
    def fn(albert, beatrice):
        return albert - beatrice

    return fn(albert=x, beatrice=y) + fn(beatrice=3, albert=7)


@specialize((int1, int2),)
def test_keywords_2(x, y):
    def fn1(x, albert, beatrice):
        return albert * (x - beatrice)

    def fn2(y, albert, beatrice):
        return y * (albert - beatrice)

    fn = fn1 if x < 0 else fn2

    return fn(5, albert=x, beatrice=y) + fn(9, albert=3, beatrice=7)


@specialize((int1, int2),)
def test_keywords_defaults(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    return fn(beatrice=x + y)


@specialize((int1, int2),)
def test_kwarg(x, y):
    def fn(albert=1, beatrice=10):
        return albert - beatrice

    def proxy(*args, **kwargs):
        return fn(*args, **kwargs)

    return proxy(x, beatrice=y), proxy(x, y), proxy(beatrice=x, albert=y)
