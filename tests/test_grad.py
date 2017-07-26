from myia.validate import grad_test as grad_test_helper
from pytest import mark, fail


xfail = mark.xfail


def grad_test(*tests):
    """
    Test the gradient of a function. This performs the following
    tests:

    1.  When applied on the given arguments, all of the following
        functions must return the same value:
        a. The original, pure Python function.
        b. The myia-compiled function.
        c. The grad-transformed myia function.

    2.  The gradient on each input, given an output gradient of
        one, must be within a small relative error of the
        finite-difference estimation of the function, using
        the pure Python implementation:
        * diff = f(x + eps, ...)

    TODO: allow specifying the gradient values explicitly.
    """
    def decorate(fn):
        try:
            exc = None
            compilation_data = grad_test_helper(fn)
            testfn = compilation_data['test']
        except Exception as e:
            exc = e

        def test(test_data):
            if exc:
                raise exc
            results = testfn(test_data)
            if not results['match']:
                for row in ['python', 'myia', 'myiag']:
                    print(f'{row}:\t{results[row]}')
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
