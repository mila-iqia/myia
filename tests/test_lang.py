
from copy import copy
from pytest import mark
from types import SimpleNamespace

from myia.api import standard_debug_pipeline

from .common import mysum, Point


lang_pipeline = standard_debug_pipeline \
    .select('parse', 'resolve', 'export')


def parse_compare(*tests):
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
            py_result = fn(*map(copy, args))
            myia_fn = lang_pipeline.run(input=fn)['output']
            myia_result = myia_fn(*map(copy, args))
            assert py_result == myia_result

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


#############
# Constants #
#############


@parse_compare(())
def test_constant():
    return 1


###################
# Some primitives #
###################


@parse_compare((1, 4), (5, -13))
def test_prim_add(x, y):
    return x + y


@parse_compare(1, -13)
def test_prim_addct(x):
    return x + 1


#############
# Variables #
#############


@parse_compare(2)
def test_parameter(x):
    return x


@parse_compare((4, 6))
def test_variable(x, y):
    z = x + y
    return z


@parse_compare((4, 6))
def test_multiple_targets(x, y):
    a, b = c = x, y
    return (a, b, c)


@parse_compare(2)
def test_multiple_variables(x):
    y = x + 1
    z = x + 2
    w = y + z
    return w + x + y + z


@parse_compare(13)
def test_shadow_variable(x):
    x = x * 2
    x = x + 7
    x = -x
    return x


c = 2


@parse_compare(5)
def test_globals(x):
    return x + c


def _f(x):
    return x + 2


@parse_compare(5)
def test_call_global(x):
    return _f(x)


@parse_compare((4, 7))
def test_swap(x, y):
    x, y = y + 3, x - 8
    return x, y


###################
# Data structures #
###################


@parse_compare(13)
def test_tuple(x):
    return x, x + 1, x + 2


@parse_compare(((1, 2, 3, 4),))
def test_getitem(x):
    return x[1]


@parse_compare((SimpleNamespace(x=5, y=2)))
def test_getattr(pt):
    return pt.x


@parse_compare((SimpleNamespace(x=5, y=2)))
def test_getattr_function(pt):
    return getattr(pt, 'x')


@parse_compare((2, 3))
def test_method(x, y):
    return x.__add__(y)


################
# if statement #
################


@parse_compare(-10, 0, 10)
def test_if(x):
    if x > 0:
        return 1
    else:
        return -1


@parse_compare(-100, -5, 5, 100, 0)
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


@parse_compare(-1, 0, 1)
def test_if2(x):
    if x > 0:
        a = 10
        b = 20
    else:
        a = 101
        b = 202
    return a + b


@parse_compare(-1, 0, 1)
def test_if3(x):
    a = 10
    b = 20
    if x > 0:
        a = 100
    return a + b


@parse_compare(1, -1)
def test_multiple_return(x):
    if x > 0:
        return 1
    return 2


@parse_compare((7, 3), (1, 3))
def test_max(x, y):
    if x > y:
        return x
    else:
        return y


@parse_compare((7, 3), (-1, 3))
def test_ifexpr(x, y):
    return x * x if x > 0 else y * y


@parse_compare((7, 3), (1, 3))
def test_max_expr(x, y):
    return x if x > y else y


@parse_compare((7, 3), (-1, 3), (-3, 1), (-1, -1))
def test_and(x, y):
    return x > 0 and y > 0


@parse_compare((7, 3), (-1, 3), (-3, 1), (-1, -1))
def test_or(x, y):
    return x > 0 or y > 0


@parse_compare((7, 3), (-1, 3), (-3, 1), (-1, -1))
def test_band(x, y):
    return (x > 0) & (y > 0)


@parse_compare((7, 3), (-1, 3), (-3, 1), (-1, -1))
def test_bor(x, y):
    return (x > 0) | (y > 0)


###################
# while statement #
###################


@parse_compare((100, 10), (50, 7))
def test_while(x, y):
    while x > 0:
        x = x - y
    return x


@parse_compare((10, 10))
def test_nested_while(x, y):
    result = 0
    i = x
    while i > 0:
        j = y
        while j > 0:
            result = result + 2
            j = j - 1
        i = i - 1
    return result


@parse_compare(10)
def test_return_in_while(x):
    while x > 0:
        x = x - 1
        return x
    return -1  # pragma: no cover


@parse_compare(10)
def test_return_in_double_while(x):
    while x > 0:
        while x > 0:
            x = x - 1
            return x
    return -1  # pragma: no cover


@parse_compare(10)
def test_if_return_in_while(x):
    while x > 0:
        if x == 5:
            return x
        else:
            x = x - 1
    return -1


#################
# for statement #
#################


@parse_compare(([1, 2, 3, 4],))
def test_for(xs):
    result = 0
    for x in xs:
        result = result + x
    return result


############
# closures #
############


@parse_compare(())
def test_nested():
    x = 2

    def g():
        return x
    return g()


@parse_compare(50)
def test_closure(x):
    def g(y):
        # Closes over x
        return x + y

    def h():
        # Closes over g
        return g(5)

    return h()


def test_closure_recur():
    # This cannot run with parse_compare since we need to reference the
    # top-level function

    def f(x, y):
        return fn(x - 1, y)

    def fn(x, y):
        def g(x):
            return x + 1
        if x == 0:
            return g(y)
        else:
            return f(x, g(y))

    py_result = fn(1, 2)
    myia_fn = lang_pipeline.run(input=fn)['output']
    myia_result = myia_fn(1, 2)
    assert py_result == myia_result


@parse_compare(())
def test_closure2():
    def g(x):
        def f():
            return x
        return f
    return g(2)()


@parse_compare(1)
def test_closure3(x):
    def g():
        def h():
            return x
        return h
    return g()()


@parse_compare(2)
def test_fn1(x):
    def g(x):
        return x
    return g(x)


@parse_compare(())
def test_fn2():
    def g(x):
        def f():
            return x
        return f()
    return g(2)


@parse_compare(())
def test_fn3():
    def g(x):
        def f():
            return x
        return f() + 1
    return g(2)


@parse_compare(())
def test_fn4():
    def g(x):
        y = x + 1

        def f():
            return y
        return f() + 1
    return g(2)


@parse_compare(())
def test_fn5():
    def g(x):
        def f(y):
            return y + 1
        return f(x + 1)
    return g(2)


###########
# Lambda #
###########


@parse_compare((5,))
def test_lambda(x):
    f = lambda y: x + y  # noqa
    return f(x)


@parse_compare((5,))
def test_lambda2(x):
    f = lambda y, z: x + y * z  # noqa
    return f(10, x)


#############
# Recursion #
#############


@parse_compare(10)
def test_rec1(x):
    def f(x):
        if x >= 0:
            return f(x - 1)
        else:
            return x
    return f(x)


#############
# MetaGraph #
#############


@parse_compare((2, 3, 4))
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


###############
# Integration #
###############


@parse_compare(2, 1)
def test_pow8(x):
    i = 0
    while i < 3:
        x = x + x
        i = i + 1
    return x


@parse_compare(2, 3)
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


@parse_compare(0, 1, 4, 10)
def test_fact(x):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(x)


@parse_compare(42)
def test_record(x):
    return Point(x, x)
