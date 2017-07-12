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

        def test(inputs, output):
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
