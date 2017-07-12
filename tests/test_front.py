from myia.front import parse_function, MyiaSyntaxError
from myia.interpret import evaluate
import pytest
import inspect

mark = pytest.mark
xfail = pytest.mark.xfail

_functions = {}


def myia_test(*tests):
    """
    Decorate a test function that is meant to be parsed by myia.

    Returns a unit test that will parse the function, and then for each
    `(inputs, output)` pair in `tests` it will check that the function
    outputs the right thing on the given inputs, and also (sanity check)
    that the pure Python, undecorated function returns that same output.

    Arguments:
    tests -- One or more (inputs, output) pair(s). `inputs` must be
             a single input or a tuple of input values that will be given
             as the argument(s) to the function. `output` must be
             the value the function should return for these inputs.
    """

    def decorate(fn):
        fname = fn.__name__

        def test(inputs, output, gradOut=None):
            if fname not in _functions:
                data = parse_function(fn)
                _functions.update(data)
            node = _functions[fname]

            if not isinstance(inputs, tuple):
                inputs = inputs,

            python_result = fn(*inputs)
            myia_result = evaluate(node, _functions)(*inputs)

            assert python_result == output
            assert myia_result == output
            # TODO:
            #assert grad_result == gradOut

        return pytest.mark.parametrize('inputs,output', list(tests))(test)

    return decorate


def myia_syntax_error(fn):
    """
    Decorate a test function that is expected to produce a
    syntax error in myia.
    """
    fname = fn.__name__

    def test():
        with pytest.raises(MyiaSyntaxError):
            parse_function(fn)

    return test


@myia_test(((1, 2), 3))
def test_just_add(x, y):
    return x + y


@myia_test((13, -33))
def test_shadow_variable(x):
    x = x * 2
    x = x + 7
    x = -x
    return x


@myia_test((-10, -1), (0, -1), (10, 1))
def test_if(x):
    if x > 0:
        return 1
    else:
        return -1


@myia_test((-1, 303), (0, 303), (1, 30))
def test_if2(x):
    if x > 0:
        a = 10
        b = 20
    else:
        a = 101
        b = 202
    return a + b


@myia_test(((100, 10), 0), ((50, 7), -6))
def test_while(x, y):
    while x > 0:
        x -= y
    return x


# ReLU test with multiple return statements
# If x > 0, then derivative is 1, else it's 0
@myia_test(
    (5, 5),
    (0, 0),
    (-1, 0)
)
def test_relu(x):
    if x > 0:
        return x
    else:
        return 0
#testGrad(relu, [5], [5], [1])
#testGrad(relu, [0], [0], [0])
#testGrad(relu, [-1], [0], [0])


# If x > y, then dx/dy = 1, else 0
@myia_test(
    ((7, 3), 7),
    ((1, 3), 3)
)
def test_max(x, y):
    if x > y:
        return x
    else:
        return y
#testGrad(max, [7, 3], [7], [1, 0])
#testGrad(max, [1, 3], [3], [0, 1])


# Computes x^8, d/dx = 8x^7
@myia_test(
    (2, 256),
    (1, 1)
)
def test_pow8(x):
    i = 0
    while i < 3:
        x = x * x
        i = i + 1
    return x
#testGrad(pow8, [2], [256], [1024])
#testGrad(pow8, [1], [1], [8])


# Test nested loops
# Computes y = x^10, d/dx=10x^9
@myia_test(
    (2, 1024)
)
def pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v
#testGrad(pow10, [2], [1024], [5120])


##########
# Errors #
##########


@myia_syntax_error
def test_stxerr_varargs1(x, y, *args):
    return 123


@myia_syntax_error
def test_stxerr_varargs2(**kw):
    return 123


##################
# Known failures #
##################


@xfail
@myia_test((50, 55))
def test_forward_ref(x):
    def h():
        return g(5)

    def g(y):
        return x + y

    return h()
