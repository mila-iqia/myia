
import pytest
from pytest import mark
import numpy as np
from types import FunctionType
from dataclasses import dataclass

from myia.abstract import from_value, AbstractJTagged, InferenceError
from myia.api import myia
from myia.pipeline import standard_resources, standard_pipeline
from myia.composite import grad, value_and_grad
from myia.debug.finite_diff import GradTester, NoTestGrad, clean_args
from myia.grad import J as realJ
from myia.pipeline import pipeline_function, PipelineDefinition, steps
from myia.pipeline.steps import Validator
from myia.prim import ops as P, Primitive
from myia.prim.py_implementations import J, scalar_add, scalar_mul, \
    array_to_scalar, scalar_to_array, array_map, array_reduce, scalar_div, \
    distribute, dot, reshape, transpose, scalar_cast
from myia.prim.py_implementations import py_registry as pyi
from myia.validate import whitelist, validate_abstract
from myia.frontends import load_frontend

from .common import f64, u64, MA, MB, to_abstract_test


@dataclass
class Point:
    x: f64
    y: f64

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


grad_whitelist = whitelist | {P.J, P.Jinv}


@validate_abstract.variant
def grad_validate_abstract(self, t: AbstractJTagged):
    pass


step_grad_validate = Validator.partial(
    whitelist=grad_whitelist,
    validate_abstract=grad_validate_abstract
)


@pipeline_function
def grad_wrap(self, graph):
    if isinstance(graph, Primitive):
        jg = realJ(graph, self.resources)
        g = grad.make_gf(jg, jg.parameters, wrt=range(len(jg.parameters)),
                         dbg=jg.debug, sens_param=True)
    else:
        g = grad.make_gf(graph, graph.parameters,
                         wrt=range(len(graph.parameters)),
                         dbg=graph.debug, sens_param=True,
                         apply_j=True)
    return g


grad_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        opt=steps.step_debug_opt,
        validate=step_grad_validate,
        export=steps.step_debug_export,
    )
)


def test_GradTester():

    def f(x, y):
        return x / y

    def df(x, y, dz):
        return dz / y, -dz * x / (y * y)

    arglist = (
        (7.3, 4.2),
        (np.array([[4.3, 2.0], [5.1, 7.7], [3.4, 8.2]]),
         np.array([[1.2, 5.0], [3.3, 2.7], [6.9, 7.2]])),
    )

    for args in arglist:
        gtest = GradTester(
            fn=f,
            gfn=df,
            args=args,
            argnames=['x', 'y'],
            outnames=['out']
        )

        gtest.assert_match()


def test_GradTester_outtup():

    def f(x, y):
        return x * y, x / y

    def df(x, y, dz):
        dz1, dz2 = dz
        return (dz1 * y + dz2 / y,
                dz1 * x + -dz2 * x / (y * y))

    gtest = GradTester(
        fn=f,
        gfn=df,
        args=(7.3, 4.2),
        argnames=['x', 'y'],
        outnames=None
    )

    gtest.assert_match()


prim_tests = {
    P.scalar_add: [(-7.1, 4.3)],
    P.scalar_sub: [(-7.1, 4.3)],
    P.scalar_mul: [(-7.1, 4.3)],
    P.scalar_div: [(-7.1, 4.3)],
    P.scalar_pow: [(7.1, 4.3), (5.3, -1.2)],
    P.scalar_uadd: [(-7.1,)],
    P.scalar_usub: [(-7.1,)],
    # P.scalar_gt: [(-7.1, 4.3)],
    # P.scalar_lt: [(-7.1, 4.3)],
    # P.scalar_ge: [(-7.1, 4.3)],
    # P.scalar_le: [(-7.1, 4.3)],
}


def _grad_test(fn, obj, args,
               sens_type=f64,
               pipeline=grad_pipeline,
               rel_error=1e-3):
    pipeline = pipeline.insert_after('parse', grad_wrap=grad_wrap)
    argspec = tuple(
        from_value(arg, broaden=True, frontend=load_frontend('numpy'))
        for arg in clean_args(args))
    sens_type = to_abstract_test(sens_type)
    if isinstance(obj, FunctionType):
        res = pipeline.run(input=obj, argspec=[*argspec, sens_type])
    else:
        pip = pipeline.configure(parse=False)
        res = pip.run(graph=obj, argspec=[*argspec, sens_type])
    gtest = GradTester(
        fn=fn,
        gfn=res['output'],
        args=args,
        argnames=[f'in{i}' for i in range(len(args))],
        outnames=None,
        rel_error=rel_error
    )
    gtest.assert_match()


@pytest.mark.parametrize('prim,cases', prim_tests.items())
def test_prim_grads(prim, cases):
    for case in cases:
        _grad_test(pyi[prim], prim, case)


def grad_test(*tests, pipeline=grad_pipeline, rel_error=1e-3):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    Arguments:
        tests: One or more inputs tuple.

    """

    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)

            _grad_test(fn, fn, args, pipeline=pipeline, rel_error=rel_error)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@grad_test((13.0, 14.0))
def test_null(x, y):
    """Test null gradient."""
    return 10.0 + 28.0 / 43.0


@grad_test((1.0, 4.0), (5.0, -13.0))
def test_grad_add(x, y):
    return x + y


@grad_test((3.0, 4.0))
def test_grad_expr(x, y):
    return x**3.0 * y**4.0


@grad_test((3.0,))
def test_constant(x):
    """Test the use of a literal in the expression."""
    return 18.0 * x


@grad_test((3.0,))
def test_dup_args_in_call(x):
    """The naive gradient update rule fails when a function's arguments
    contain the same variable more than once."""
    return x * x


@grad_test((3.0,))
def test_quadruple_args_in_call(x):
    """Test that duplicated arguments still cause no problem even if
    there are four of them."""
    def g(a, b, c, d):
        return a * b * c * d
    return g(x, x, x, x)


@grad_test((3.0, 5.0))
def test_tuples(x, y):
    tup = x + y, x * y
    z = tup[0] + tup[1]
    return z


@grad_test((Point(3.0, 5.0),), pipeline=standard_pipeline)
def test_dataclass(pt):
    return pt.x * pt.y


@grad_test((Point(3.0, 5.0),), pipeline=standard_pipeline)
def test_dataclass_2(pt):
    return pt.abs()


@grad_test((4.0, 5.0))
def test_hof(a, b):
    """Test higher order functions."""
    def f(g, x):
        return g(x) * g(x + 10.0)

    def g(x):
        return x * b

    return f(g, a) + f(g, b)


@grad_test((4.0, 5.0))
def test_hof_tup(a, b):
    """Test higher order functions."""
    def f(gh, x, y):
        g, h = gh
        # return g(x, y) * h(x, y)
        return scalar_mul(g(x, y), h(x, y))

    return f((scalar_add, scalar_mul), a, b)


@grad_test((4.0, 5.0))
def test_simple_closure(a, b):
    """Test some trivial closures."""
    def f():
        return a + 1.0

    def g():
        return b + 2.0
    return f() * g()


@grad_test((4.0,))
def test_closure(a):
    """This is the closure test in the paper."""
    def x1(b):

        def x4(c):
            return b
        return x4
    x2 = x1(a)
    x3 = x2(1.0)
    return x3


@grad_test((4.0, 5.0), (6.4, -7.8))
def test_if(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a
    else:
        return b


@grad_test((4.0, 5.0), (6.4, -7.8))
def test_if2(a, b):
    if a > b:
        return a * a
    else:
        return b + b


@grad_test(4.1,)
def test_fact(x):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(x)


@grad_test((4.1,), pipeline=standard_pipeline)
def test_fact_opt(x):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(x)


@grad_test((4.0,))
def test_while(x):
    rval = x
    while rval < 100:
        rval = rval * rval
    return rval


@grad_test((4.0, 5.0, 2.0),)
def test_while_2(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval = rval + y
        x = x - z
    return rval


@grad_test(2.0,)
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


@grad_test(([1.0, 2.0, 3.0, 4.0],),)
def test_list_while(xs):
    y = 1.0
    index = 0
    while index < len(xs):
        y = y * xs[index]
        index = index + 1
    return y


@grad_test(([1.0, 2.0, 3.0, 4.0],), pipeline=standard_pipeline)
def test_list_for(xs):
    y = 1
    for x in xs:
        y = y * x
    return y


@grad_test(4.5,)
def test_nested_closure(x):
    a = x * x

    def f():
        b = x + 5

        def g():
            return a + b

        def h():
            return a * b
        return g if x < 0 else h
    return f()()


@grad_test((4.5, 6.7),)
def test_functions_in_tuples(x, y):
    tup = scalar_add, scalar_mul
    f, g = tup
    return f(x, y) + g(x, y)


@mark.xfail(reason="A DummyInferrer ends up being called")
@grad_test((4.5, 6.7),)
def test_closures_in_tuples(x, y):
    def f():
        return x * y

    def g():
        return x + y

    tup = f, g
    ff, gg = tup
    return ff() + gg()


@grad_test((MA(2, 3), MB(2, 3)),)
def test_array_operations(xs, ys):
    div = array_map(scalar_div, xs, ys)
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((3.1, 7.6),)
def test_array_operations_distribute(x, y):
    xs = distribute(scalar_to_array(x), (4, 3))
    ys = distribute(scalar_to_array(y), (4, 3))
    div = array_map(scalar_div, xs, ys)
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((MA(2, 3), MB(2, 3)),)
def test_array_operations_reshape(xs, ys):
    xs = reshape(xs, (6,))
    ys = reshape(ys, (6,))
    div = array_map(scalar_div, xs, ys)
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((MA(2, 3), MB(2, 3)),)
def test_array_operations_std(xs, ys):
    div = xs / ys
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((MA(2, 3), MB(3, 4)),)
def test_dot(x, y):
    d = dot(x, y)
    sm = array_reduce(scalar_add, d, ())
    return array_to_scalar(sm)


@grad_test((MA(3, 4), MB(2, 3)),)
def test_transpose(x, y):
    xt = transpose(x, (1, 0))
    yt = transpose(y, (1, 0))
    d = dot(xt, yt)
    sm = array_reduce(scalar_add, d, ())
    return array_to_scalar(sm)


@grad_test((MA(3, 4), MB(2, 3), NoTestGrad(1), NoTestGrad(0)),)
def test_transpose2(x, y, axis1, axis2):
    perm = (scalar_cast(axis1, u64),
            scalar_cast(axis2, u64))
    xt = transpose(x, perm)
    yt = transpose(y, perm)
    d = dot(xt, yt)
    sm = array_reduce(scalar_add, d, ())
    return array_to_scalar(sm)


def _runwith(f, *args):
    argspec = tuple(
        from_value(arg, broaden=True, frontend=load_frontend('numpy'))
        for arg in args)
    res = grad_pipeline.run(input=f, argspec=argspec)
    return res['output'](*args)


def test_freevar_outside_grad():

    def f(x, y):
        a = x * x

        def mula(z):
            return a * z
        _, bprop = J(mula)(J(y))
        return bprop(1)[1]

    assert _runwith(f, 5.0, 8.0) == 25.0


def test_freegraph_outside_grad():

    def f(x, y):
        def sqx():
            return x * x

        def mulsqx(z):
            return sqx() * z
        _, bprop = J(mulsqx)(J(y))
        return bprop(1)[1]

    assert _runwith(f, 5.0, 8.0) == 25.0


def test_grad_interface():
    def f(x, y):
        a = x ** 3
        b = y ** 4
        return a * b

    @myia
    def grads(x, y):
        return (
            grad(f, 'x')(x, y),
            grad(f, 0)(x, y),
            grad(f, 'y')(x, y),
            grad(f, 'x', 'y')(x, y),
            grad(f, '*')(x, y),
            value_and_grad(f)(x, y),
        )

    @myia
    def gradbad(x, y):
        return grad(f, (0, 1))(x, y)

    @myia
    def gradbad2(x, y):
        return grad(f, 'z')(x, y)

    @myia
    def gradbad3(x, y, z):
        return grad(f, z)(x, y)

    x, y = 2.0, 3.0

    dx = 3 * (x ** 2) * (y ** 4)
    dy = 4 * (y ** 3) * (x ** 3)

    assert grads(x, y) == (dx, dx, dy, (dx, dy), (dx, dy), (f(x, y), dx))

    with pytest.raises(InferenceError):
        print(gradbad(x, y))

    with pytest.raises(InferenceError):
        print(gradbad2(x, y))

    with pytest.raises(InferenceError):
        print(gradbad3(x, y, 0))
