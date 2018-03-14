
from pytest import mark, fail
from copy import copy

from myia.api import parse, compile
from myia.grad import grad
from myia.debug.finite_diff import GradTester
from myia.py_implementations import J, tail


def grad_test(*tests):
    """
    Test the gradient of a function. This performs the following
    tests:
    1.  When applied on the given arguments, all of the following
        function calls must return the same value:
        a. The original, pure Python function.
        b. The myia-compiled function.
        c. The grad-transformed myia function.
    2.  The gradient on each input, given an output gradient of
        one, must be within a small error of the symmetric
        finite-difference estimation:
        * diff_dx = f(..., x + eps, ...) - f(..., x - eps, ...) / 2eps
        * computed_dx = J(f)(..., x, ...)[1](1)
        * error_dx = abs(diff_dx - computed_dx) > eps_2

    TODO: allow specifying the gradient values explicitly.
    """

    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)
            # TODO: avoid re-parsing and re-gradding every time
            graph = parse(fn)
            gfn = compile(grad(graph))
            py_result = fn(*map(copy, args))
            myia_result, bprop_fn = gfn(*map(copy, args))
            assert py_result == myia_result

            argnames = [p.debug.name for p in graph.parameters]

            test = GradTester(fn=fn,
                              gfn=bprop_fn,
                              args=args,
                              argnames=argnames,
                              outnames=[fn.__name__])
            results = test.compare()

            for arg, d in results.items():
                if not d['match']:
                    print(f'Argument {arg}:')
                    print(f'\tFinite differences: {d["difference"]}')
                    print(f'\tGradient output:    {d["exact"]}')
                    fail(f'Mismatch in gradients for {arg} (see stdout)')

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@grad_test((13, 14))
def test_null(x, y):
    """Test null gradient."""
    return 10 + 28 / 43


@grad_test((3, 4))
def test_tuple(x, y):
    """Test multiple outputs."""
    return (x + y, x - y, x * y, x / y)


@grad_test((3, 4, 5))
def test_expression(x, y, z):
    """Test a more complex expression."""
    return x * y + y / z


@grad_test((3, 4, 5))
def test_variables(x, y, z):
    """Test an expression with temporary variables."""
    a = x * y
    b = y * a
    c = a + b
    return c / z


@grad_test((3, 4, 5))
def test_shadowing(x, y, z):
    """Test an expression where variables are shadowed."""
    x = x * y
    y = y * z
    z = x * z
    return x + y + z


@grad_test((3,))
def test_constant(x):
    """Test the use of a literal in the expression."""
    return 18 * x


@grad_test((3,))
def test_dup_args_in_call(x):
    """The naive gradient update rule fails when a function's arguments
    contain the same variable more than once."""
    return x * x


@grad_test((3,))
def test_quadruple_args_in_call(x):
    """Test that duplicated arguments still cause no problem even if
    there are four of them."""
    def g(a, b, c, d):
        return a * b * c * d
    return g(x, x, x, x)


@grad_test((3, 5))
def test_tuples(x, y):
    tup = x + y, x * y
    z = tup[0] + tup[1]
    return z


@grad_test((4, 5))
def test_simple_closure(a, b):
    """Test some trivial closures."""
    def f():
        return a + 1

    def g():
        return b + 2
    return f() * g()


@grad_test((4,))
def test_closure(a):
    """This is the closure test in the paper."""
    def x1(b):

        def x4(c):
            return b
        return x4
    x2 = x1(a)
    x3 = x2(1)
    return x3


# TODO: test when a == b (finite diff won't return anything sensible)
@grad_test((4, 5), (68, -4))
def test_if(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a * b
    else:
        return b + a


@grad_test((4, 5, 2), (7, 3, 1))
def test_while(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval += y
        x -= z
    return rval


# @grad_test((2,), (4,), (8,))
# def test_for(n):
#     y = 0
#     for x in range(10):
#         y += n
#     return y


@grad_test((2,), (3,))
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


@grad_test(3)
def test_grad2_simple(x):
    def f(x):
        return x * x * x * x
    jf = J(f)
    tup = jf(x)
    bprop = tup[1]
    all_res = bprop(1)
    res = tail(all_res)
    return res


@grad_test((4, 5, 2), (-7, 3, 1))
def test_grad2_if(x, y, z):
    def f(x, y, z):
        if x < 0:
            return y * z
        else:
            return y + z
    jf = J(f)
    tup = jf(x, y, z)
    bprop = tup[1]
    all_res = bprop(1)
    res = tail(all_res)
    return res


@grad_test((4, 5, 2), (7, 3, 1))
def test_grad2_while(x, y, z):
    def f(x, y, z):
        rval = 0
        # Cannot compare to 0 or finite diff is unstable
        while x > -0.1:
            rval += y
            x -= z
        return rval
    jf = J(f)
    tup = jf(x, y, z)
    bprop = tup[1]
    all_res = bprop(1)
    res = tail(all_res)
    return res


@grad_test(3)
def test_grad2_pow(x):
    def f(x):
        v = x
        j = 0
        while j < 3:
            i = 0
            while i < 3:
                v = v * x
                i = i + 1
            j = j + 1
        return v
    jf = J(f)
    tup = jf(x)
    bprop = tup[1]
    all_res = bprop(1)
    res = tail(all_res)
    return res
