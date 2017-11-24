"""
Test the forward mode of Myia functions (no gradients).
"""

from myia.parse import MyiaSyntaxError, parse_function
from myia.front import compile
from myia.stx import Symbol
import pytest

mark = pytest.mark
xfail = pytest.mark.xfail


def myia_test(*tests):
    """
    Decorate a test function that is meant to be parsed by myia.

    Returns a unit test that will parse the function, and then for each
    `(inputs, output)` pair in `tests` it will check that the function
    outputs the right thing on the given inputs, and also (sanity check)
    that the pure Python, undecorated function returns that same output.

    Arguments:
        tests: One or more (inputs, output) pair(s). `inputs` must be
               a single input or a tuple of input values that will be
               given as the argument(s) to the function. `output` must
               be the value the function should return for these inputs.
    """

    def decorate(fn):
        def test(inputs, output, gradOut=None):
            node = parse_function(fn)

            if not isinstance(inputs, tuple):
                inputs = inputs,

            python_result = fn(*inputs)
            myia_result = compile(node)(*inputs)

            assert python_result == output
            assert myia_result == output
            # TODO:
            #assert grad_result == gradOut

        m = pytest.mark.parametrize('inputs,output', list(tests))(test)
        m.__orig__ = fn
        return m

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


############
# Features #
############


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


@myia_test((-100, 1), (-5, 2), (5, 3), (100, 4), (0, 5))
def test_nested_if(x):
    if x < 0:
        if x < -10:
            return 1
        else:
            return 2
    elif x > 0:
        if x < 10:
            return 3
        else:
            return 4
    else:
        return 5


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


@myia_test(((10, 10), 200))
def test_nested_while(x, y):
    result = 0
    i = x
    # Fails if this line is removed, see test_while_blockvar for why
    j = 0
    while i > 0:
        j = y
        while j > 0:
            result += 2
            j -= 1
        i -= 1
    return result


@myia_test(((10, 20), 10))
def test_swap(x, y):
    y, x = x, y
    return x - y


@myia_test((50, 55))
def test_closure(x):
    def g(y):
        # Closes over x
        return x + y

    def h():
        # Closes over g
        return g(5)

    return h()


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
    (2, 1024),
    (3, 59049)
)
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
#testGrad(pow10, [2], [1024], [5120])


# Test for loops
@myia_test(
    (2, 1),
    (10, 45)
)
def test_for(n):
    y = 0
    for x in range(n):
        y += x
    return y


@myia_test(
    (((),), 0),
    (((54,),), 54),
    (((1, 2, 3, 4),), 10)
)
def test_sum(xs):
    y = 0
    for x in xs:
        y += x
    return y


@myia_test(
    (((),), 0),
    (((54,),), 1),
    (((1, 2, 3, 4),), 16)
)
def test_nested_for(xs):
    y = 0
    for x1 in xs:
        for x2 in xs:
            y += 1
    return y


##########
# Errors #
##########


@myia_syntax_error
def test_stxerr_varargs1(x, y, *args):
    "We do not support *args"
    return 123


@myia_syntax_error
def test_stxerr_varargs2(**kw):
    "We do not support **kw"
    return 123


@myia_syntax_error
def test_stxerr_kwargs(x):
    "We do not support keyword arguments in function calls"
    return range(start = x, end = x)


@myia_syntax_error
def test_stxerr_function_slice(x):
    "We do not support setting a slice directly on the result of a call"
    (x + x)[0] = 1
    return x


##################
# Known failures #
##################


@xfail
@myia_test((50, 55))
def test_forward_ref(x):
    def h():
        # h does not have g in its closure, because it is
        # defined before g
        return g(5)

    def g(y):
        return x + y

    return h()


@xfail
@myia_test((10, 0))
def test_while_blockvar(x):
    while x > 0:
        # The local block variable y causes the bug, because myia
        # requires it to be set outside the block so that it is
        # guaranteed to have a value if 0 loops are executed. This is
        # what we want if y is referenced outside the loop, but it
        # isn't in this case, so we want to fix this eventually.
        y = x * x
        x -= y
    return x
