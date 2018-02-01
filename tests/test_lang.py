
from pytest import mark
from types import SimpleNamespace
from myia.api import parse, run
from copy import copy


def parse_compare(*tests):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for each `inputs`
    tuple in `tests` it will check that the pure Python, undecorated function
    returns that same output.

    Arguments:
        tests: One or more inputs tuple.
    """

    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)
            # TODO: avoid re-parsing every time
            fn2 = parse(fn)
            py_result = fn(*map(copy, args))
            myia_result = run(fn2, tuple(map(copy, args)))
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


@parse_compare(5)
def test_augassign(x):
    x += 3
    x *= 8
    return x


@parse_compare((4, 7))
def test_swap(x, y):
    x, y = y + 3, x - 8
    return x, y


@parse_compare(([1, 2, 3], 1))
def test_setitem(x, y):
    x[y] = 21
    return x


@parse_compare(([1, 2, 3], 1))
def test_augsetitem(x, y):
    x[y] += 21
    return x


@parse_compare((SimpleNamespace(x=5, y=2)))
def test_setattr(pt):
    pt.x = 3
    pt.y = 21
    return pt


@parse_compare((SimpleNamespace(x=5, y=2)))
def test_augsetattr(pt):
    pt.x += 4
    return pt


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


###############
# Integration #
###############


@parse_compare(2, 1)
def test_pow8(x):
    i = 0
    while i < 3:
        x *= x
        i += 1
    return x


@parse_compare(2, 3)
def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v *= x
            i += 1
        j += 1
    return v


@parse_compare(0, 1, 4, 10)
def test_fact(x):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(x)


##########
# Issues #
##########


@mark.xfail(reason='The return is not triggered (#29)')
@parse_compare(10)
def test_return_in_while(x):
    while x > 0:
        x = x - 1
        return x
    return -1


@mark.xfail(reason='The return is not triggered (#29)')
@parse_compare(10)
def test_return_in_double_while(x):
    while x > 0:
        while x > 0:
            x = x - 1
            return x
    return -1
