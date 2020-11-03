from pytest import mark

from myia.pipeline import scalar_debug_pipeline, steps
from myia.testing.common import Point, mysum
from myia.testing.multitest import mt, run, run_debug

lang_pipeline = scalar_debug_pipeline.with_steps(
    steps.step_parse,
    steps.step_resolve,
    steps.step_llift,
    steps.step_debug_export,
)


run_lang = run.configure(pipeline=lang_pipeline, backend=False)


#############
# Constants #
#############


@run_lang()
def test_constant():
    return 1


###################
# Some primitives #
###################


@mt(run_lang(1, 4), run_lang(5, -13))
def test_prim_add(x, y):
    return x + y


@mt(run_lang(1), run_lang(-13))
def test_prim_addct(x):
    return x + 1


#############
# Variables #
#############


@run_lang(2)
def test_parameter(x):
    return x


@run_lang(4, 6)
def test_variable(x, y):
    z = x + y
    return z


@run_lang(4, 6)
def test_multiple_targets(x, y):
    a, b = c = x, y
    return (a, b, c)


@run_lang(2)
def test_multiple_variables(x):
    y = x + 1
    z = x + 2
    w = y + z
    return w + x + y + z


@run_lang(13)
def test_shadow_variable(x):
    x = x * 2
    x = x + 7
    x = -x
    return x


c = 2


@run_lang(5)
def test_globals(x):
    return x + c


def _f(x):
    return x + 2


@run_lang(5)
def test_call_global(x):
    return _f(x)


@run_lang(4, 7)
def test_swap(x, y):
    x, y = y + 3, x - 8
    return x, y


###################
# Data structures #
###################


@mark.skip(reason="This test requires the inference step")
@run_lang(13)
def test_list(x):
    return [x, x + 1, x + 2]


@run_debug(2)
def test_dict(x):
    return {"x": x}


@run_lang(13)
def test_tuple(x):
    return x, x + 1, x + 2


@run_lang((1, 2, 3, 4))
def test_getitem(x):
    return x[1]


@run_debug(Point(x=5, y=2))
def test_getattr(pt):
    return pt.x


@run_debug(Point(x=5, y=2))
def test_getattr_function(pt):
    return getattr(pt, "x")


@run_debug(2, 3)
def test_method(x, y):
    return x.__add__(y)


################
# if statement #
################


@mt(run_lang(-10), run_lang(0), run_lang(10))
def test_if(x):
    if x > 0:
        return 1
    else:
        return -1


@mt(run_lang(-100), run_lang(-5), run_lang(5), run_lang(100), run_lang(0))
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


@mt(run_lang(-1), run_lang(0), run_lang(1))
def test_if2(x):
    if x > 0:
        a = 10
        b = 20
    else:
        a = 101
        b = 202
    return a + b


@mt(run_lang(-1), run_lang(0), run_lang(1))
def test_if3(x):
    a = 10
    b = 20
    if x > 0:
        a = 100
    return a + b


@mt(run_lang(1), run_lang(-1))
def test_multiple_return(x):
    if x > 0:
        return 1
    return 2


@mt(run_lang(7, 3), run_lang(1, 3))
def test_max(x, y):
    if x > y:
        return x
    else:
        return y


@mt(run_lang(7, 3), run_lang(-1, 3))
def test_ifexpr(x, y):
    return x * x if x > 0 else y * y


@mt(run_lang(7, 3), run_lang(1, 3))
def test_max_expr(x, y):
    return x if x > y else y


@mt(run_lang(7, 3), run_lang(-1, 3), run_lang(-3, 1), run_lang(-1, -1))
def test_and(x, y):
    return x > 0 and y > 0


@mt(run_lang(7, 3), run_lang(-1, 3), run_lang(-3, 1), run_lang(-1, -1))
def test_or(x, y):
    return x > 0 or y > 0


@mt(run_lang(7, 3), run_lang(-1, 3), run_lang(-3, 1), run_lang(-1, -1))
def test_band(x, y):
    return (x > 0) & (y > 0)


@mt(run_lang(7, 3), run_lang(-1, 3), run_lang(-3, 1), run_lang(-1, -1))
def test_bor(x, y):
    return (x > 0) | (y > 0)


###################
# while statement #
###################


@mt(run_lang(100, 10), run_lang(50, 7))
def test_while(x, y):
    while x > 0:
        x = x - y
    return x


@run_lang(10, 10)
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


@run_lang(10)
def test_return_in_while(x):
    while x > 0:
        x = x - 1
        return x
    return -1


@run_lang(10)
def test_return_in_double_while(x):
    while x > 0:
        while x > 0:
            x = x - 1
            return x
    return -1


@run_lang(10)
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


@run_debug((1, 2, 3, 4))
def test_for(xs):
    result = 0
    for x in xs:
        result = result + x
    return result


############
# closures #
############


@run_lang()
def test_nested():
    x = 2

    def g():
        return x

    return g()


@run_lang(50)
def test_closure(x):
    def g(y):
        # Closes over x
        return x + y

    def h():
        # Closes over g
        return g(5)

    return h()


def test_closure_recur():
    # This cannot run with run_lang since we need to reference the
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
    myia_fn = lang_pipeline(input=fn)["output"]
    myia_result = myia_fn(1, 2)
    assert py_result == myia_result


@run_lang()
def test_closure2():
    def g(x):
        def f():
            return x

        return f

    return g(2)()


@run_lang(1)
def test_closure3(x):
    def g():
        def h():
            return x

        return h

    return g()()


@run_lang(7)
def test_closure4(x):
    def f():
        return x

    def g(y):
        def h():
            return y * f()

        return h()

    return g(5)


@run_lang(2)
def test_fn1(x):
    def g(x):
        return x

    return g(x)


@run_lang()
def test_fn2():
    def g(x):
        def f():
            return x

        return f()

    return g(2)


@run_lang()
def test_fn3():
    def g(x):
        def f():
            return x

        return f() + 1

    return g(2)


@run_lang()
def test_fn4():
    def g(x):
        y = x + 1

        def f():
            return y

        return f() + 1

    return g(2)


@run_lang()
def test_fn5():
    def g(x):
        def f(y):
            return y + 1

        return f(x + 1)

    return g(2)


###########
# Lambda #
###########


@run_lang(5)
def test_lambda(x):
    f = lambda y: x + y  # noqa
    return f(x)


@run_lang(5)
def test_lambda2(x):
    f = lambda y, z: x + y * z  # noqa
    return f(10, x)


#############
# Recursion #
#############


@run_lang(10)
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


@run_debug(2, 3, 4)
def test_multitype(x, y, z):
    return mysum(x) * mysum(x, y) * mysum(x, y, z)


###############
# Integration #
###############


@mt(run_lang(2), run_lang(1))
def test_pow8(x):
    i = 0
    while i < 3:
        x = x + x
        i = i + 1
    return x


@run_lang(42)
def test_record(x):
    return Point(x, x)
