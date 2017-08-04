from myia.validate import analysis
from pytest import mark, fail


xfail = mark.xfail


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
        try:
            exc = None
            testfn = analysis('grad', fn)
        except Exception as e:
            exc = e

        def test(test_data):
            if exc:
                raise exc
            results = testfn(test_data)
            print(results)
            if not results['match']:
                for row in ['python', 'myia', 'myiag']:
                    print(f'{row}:\t{results[row+"_result"]}')
                fail('Mismatch is output values (see stdout)')
            for arg, d in results["derivatives"].items():
                if not d['match']:
                    print(f'Argument {arg}:')
                    print(f'\tFinite differences: {d["difference"]}')
                    print(f'\tGradient output:    {d["computed"]}')
                    fail(f'Mismatch in gradients for {arg} (see stdout)')

        return mark.parametrize('test_data', list(tests))(test)
    return decorate


@grad_test((3, 4))
def test_add(x, y):
    """Test the gradient of addition alone."""
    return x + y


@grad_test((3, 4))
def test_subtract(x, y):
    """Test the gradient of subtraction alone."""
    return x - y


@grad_test((3, 4))
def test_multiply(x, y):
    """Test the gradient of multiplication alone."""
    return x * y


@grad_test((3, 4))
def test_divide(x, y):
    """Test the gradient of division alone."""
    return x / y


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
    return x * 18


@xfail
@grad_test((3,))
def test_dup_args_in_call(x):
    """The naive gradient update rule fails when a function's arguments
    contain the same variable more than once."""
    # TODO: VERY VERY IMPORTANT TO FIX
    return x * x


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


# TODO: test when a == b (finite diff will return ~0.5, but not our method)
@grad_test((4, 5), (68, -4))
def test_if(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a
    else:
        return b


@grad_test((4, 5, 2), (7, 3, 1))
def test_while(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval += y
        x -= z
    return rval


@grad_test((2,), (3,))
def test_pow10(x):
    v = x
    i = 0
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v
