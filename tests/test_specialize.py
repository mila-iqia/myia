
import numpy
from pytest import mark

from myia.api import scalar_debug_pipeline, standard_debug_pipeline
from myia.composite import list_map
from myia.debug.label import short_labeler as lbl
from myia.debug.traceback import print_inference_error
from myia.infer import InferenceError
from myia.prim.py_implementations import \
    hastype, partial, scalar_add, scalar_sub, \
    scalar_usub, scalar_uadd, switch
from myia.validate import ValidationError

from .common import mysum, i64, f64


specialize_pipeline = scalar_debug_pipeline \
    .select('parse', 'infer', 'specialize', 'prepare', 'validate', 'export') \
    .configure(
        {'inferrer.tracks.value.max_depth': 1,
         'inferrer.tied_tracks': {'type': ['shape']}}
    )


specialize_pipeline_std = standard_debug_pipeline \
    .select('parse', 'infer', 'specialize',
            'prepare', 'opt', 'validate', 'export', 'wrap') \
    .configure(
        {'inferrer.tracks.value.max_depth': 1,
         'inferrer.tied_tracks': {'type': ['shape']}}
    )


def specializer_decorator(pipeline):
    def specialize(*arglists):

        def decorate(fn):
            def run_test(args):
                pip = pipeline.make()
                argspec = tuple({'value': arg} for arg in args)
                pip.resources.inferrer.fill_in(argspec)
                print(argspec)
                for arg in argspec:
                    del arg['value']

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
                        print(f'   {nlbl} ({type(n).__name__}) :: {n.type}')
                        print(f'      {err.args[0]}')
                    raise verr
                except:
                    raise

                result_final = res['output'](*args)
                if isinstance(result_py, numpy.ndarray):
                    assert (result_py == result_final).all()
                else:
                    assert result_py == result_final

            m = mark.parametrize('args', arglists)(run_test)
            m.__orig__ = fn
            return m

        return decorate
    return specialize


specialize = specializer_decorator(specialize_pipeline)
specialize_std = specializer_decorator(specialize_pipeline_std)


int1 = 13
int2 = 21

fp1 = 2.7
fp2 = 6.91


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


@specialize(([fp1, fp2],))
def test_list_map(xs):
    def square(x):
        return x * x

    return list_map(square, xs)


@specialize(([fp1, fp2], [int1, int2]))
def test_list_map_polymorphic(xs, ys):
    def square(x):
        return x * x

    return list_map(square, xs), list_map(square, ys)


@mark.xfail(reason="Cannot specialize f")
@specialize((True, [fp1, fp2], [int1, int2]))
def test_list_map_polymorphic_2(c, xs, ys):
    def square(x):
        return x * x

    def double(x):
        return x + x

    if c:
        f = square
    else:
        f = double

    return list_map(f, xs), list_map(f, ys)


@specialize((int1, int2))
def test_unused_parameter(x, y):
    return x * x


@specialize((int1,))
def test_unused_function_parameter(x):
    # The type of square will be Problem(DEAD), but that's not really an issue
    # because it is indeed not used, and we can simply replace the reference
    # by a dummy.
    def square(x):
        return x * x

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


@mark.xfail(reason="Distinct contexts are created for 2 and 3, "
                   "leading to Problem(POLY).")
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
