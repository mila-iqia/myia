
import pytest
import numpy as np
from types import FunctionType

from myia import api
from myia.api import standard_resources, Optimizer, Validator
from myia.composite import grad
from myia.debug.finite_diff import GradTester
from myia.dtype import JTagged
from myia.dshape import NOSHAPE
from myia.grad import J as realJ
from myia.opt import lib as optlib, CSE
from myia.pipeline import pipeline_function, PipelineDefinition
from myia.prim import ops as P, Primitive
from myia.prim.py_implementations import J, scalar_add, scalar_mul, typeof, \
    array_to_scalar, scalar_to_array, array_map, array_reduce, scalar_div, \
    distribute, dot
from myia.prim.py_implementations import py_implementations as pyi
from myia.validate import whitelist, validate_type

from .common import f64


A = np.array([[1.7, -6.3,  8.1],
              [5.4,  2.1, -3.3]])

B = np.array([[3.2, -8.1, -5.5],
              [0.5,  4.0,  7.9]])

C = np.array([[1.4,  5.3, -8.6, -9.9],
              [4.5,  1.0,  7.4,  6.5],
              [4.1, -3.0,  3.1,  2.2]])


grad_whitelist = whitelist | {P.J, P.Jinv}


@validate_type.variant
def grad_validate_type(self, t: JTagged):
    pass


step_grad_opt = Optimizer.partial(
    phases=dict(
        main=[
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            optlib.inline_core,
            optlib.simplify_partial,
            optlib.elim_identity,
        ],
        grad=[
            optlib.expand_J,
        ],
        renormalize='renormalize',
        elimj=[
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
        ],
        cse=CSE.partial(report_changes=False),
    )
)


step_grad_validate = Validator.partial(
    whitelist=grad_whitelist,
    validate_type=grad_validate_type
)


@pipeline_function
def grad_wrap(self, graph):
    if isinstance(graph, Primitive):
        jg = realJ(graph, self.resources)
        g = grad.make_gf(jg, jg.parameters,
                         dbg=jg.debug, sens_param=True, get_all=True)
    else:
        g = grad.make_gf(graph, graph.parameters,
                         dbg=graph.debug, sens_param=True, get_all=True,
                         apply_j=True)
    return g


grad_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=api.step_parse,
        grad_wrap=grad_wrap,
        resolve=api.step_resolve,
        infer=api.step_infer,
        specialize=api.step_specialize,
        opt=step_grad_opt,
        validate=step_grad_validate,
        export=api.step_debug_export,
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


def _grad_test(fn, obj, args, sens_type=f64):
    in_types = [{'type': typeof(arg),
                 'shape': getattr(arg, 'shape', NOSHAPE)} for arg in args]
    sens_type = {'type': sens_type}
    if isinstance(obj, FunctionType):
        res = grad_pipeline.run(input=obj, argspec=[*in_types, sens_type])
    else:
        pip = grad_pipeline.configure(parse=False)
        res = pip.run(graph=obj, argspec=[*in_types, sens_type])
    gtest = GradTester(
        fn=fn,
        gfn=res['output'],
        args=args,
        argnames=[f'in{i}' for i in range(len(args))],
        outnames=None
    )
    gtest.assert_match()


@pytest.mark.parametrize('prim,cases', prim_tests.items())
def test_prim_grads(prim, cases):
    for case in cases:
        _grad_test(pyi[prim], prim, case)


def grad_test(*tests):
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

            _grad_test(fn, fn, args)

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
        return g(x, y) * h(x, y)

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


@grad_test(4.5,)
def test_nested_closure(x):
    a = x * x
    b = x + 5

    def f():
        def g():
            return a + b
        return g
    return f()()


@grad_test((4.5, 6.7),)
def test_functions_in_tuples(x, y):
    tup = scalar_add, scalar_mul
    f, g = tup
    return f(x, y) + g(x, y)


@pytest.mark.xfail(
    reason="A DummyInferrer is unfortunately propagated into a call."
)
@grad_test((4.5, 6.7),)
def test_closures_in_tuples(x, y):
    def f():
        return x * y

    def g():
        return x + y

    tup = f, g
    ff, gg = tup
    return ff() + gg()


@grad_test((A, B),)
def test_array_operations(xs, ys):
    div = array_map(scalar_div, xs, ys)
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((3.1, 7.6),)
def test_array_operations2(x, y):
    xs = distribute(scalar_to_array(x), (4, 3))
    ys = distribute(scalar_to_array(y), (4, 3))
    div = array_map(scalar_div, xs, ys)
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((A, B),)
def test_array_operations_std(xs, ys):
    div = xs / ys
    sm = array_reduce(scalar_add, div, ())
    return array_to_scalar(sm)


@grad_test((A, C),)
def test_dot(x, y):
    d = dot(x, y)
    sm = array_reduce(scalar_add, d, ())
    return array_to_scalar(sm)


def _runwith(f, *args):
    in_types = [{'type': typeof(arg)} for arg in args]
    pip = grad_pipeline.configure(grad_wrap=False)
    res = pip.run(input=f, argspec=in_types)
    return res['output'](*args)


def test_freevar_outside_grad():

    def f(x, y):
        a = x * x

        def mula(z):
            return a * z
        res, bprop = J(mula)(J(y))
        return bprop(1)[1]

    assert _runwith(f, 5.0, 8.0) == 25.0


def test_freegraph_outside_grad():

    def f(x, y):
        def sqx():
            return x * x

        def mulsqx(z):
            return sqx() * z
        res, bprop = J(mulsqx)(J(y))
        return bprop(1)[1]

    assert _runwith(f, 5.0, 8.0) == 25.0
