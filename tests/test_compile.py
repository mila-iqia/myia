from pytest import mark
from copy import copy

from myia.api import standard_pipeline
from myia.prim.py_implementations import typeof

from myia.compile.transform import LIN_IMPLS

nnvm_enabled = 'nnvm' in LIN_IMPLS

compile_pipeline = standard_pipeline
nnvm_pipeline = compile_pipeline.configure({'compile.linear_impl': 'nnvm'})


def parse_compare(*tests, nnvm=False):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    This uses the full myia pipeline.

    Arguments:
        tests: One or more inputs tuple.

    """
    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)
            py_result = fn(*map(copy, args))
            argspec = tuple({'type': typeof(a)} for a in args)
            if nnvm:
                myia_fn = nnvm_pipeline.run(input=fn,
                                            argspec=argspec)['output']
            else:
                myia_fn = compile_pipeline.run(input=fn,
                                               argspec=argspec)['output']
            myia_result = myia_fn(*map(copy, args))
            assert py_result == myia_result

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


if nnvm_enabled:
    @parse_compare((2, 3), nnvm=True)
    def test_simple(x, y):
        return x + y

    @parse_compare((2,), nnvm=True)
    def test_simple2(x):
        return 44 - x


@parse_compare((2, 3))
def test_simple(x, y):
    return x + y


@parse_compare((42,))
def test_constant(x):
    return x == 42


@parse_compare((33, 42), (42, 33))
def test_if(x, y):
    if x > y:
        return x - y
    else:
        return y - x


@parse_compare((33, 42), (44, 42))
def test_if_nottail(x, y):
    def cap(x):
        if x > 42:
            x = 42
        return x
    return y - cap(x)


@parse_compare((42, 33))
def test_call(x, y):
    def f(x):
        return x * x

    return f(x) + f(y)


@parse_compare((42,))
def test_tailcall(x):
    def fact(x, a):
        if x == 1:
            return a
        else:
            return fact(x - 1, a * x)
    return fact(x, 1)


@parse_compare((-1,), (1,))
def test_callp(x):
    def fn(f, x):
        return f(x)

    def f(x):
        return -x

    return fn(f, -42)


@parse_compare((True, 42, 33))
def test_call_hof(c, x, y):
    def f1(x, y):
        return x + y

    def f2(x, y):
        return y + x

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, 2) + choose(not c)(2, y)
    
