import pytest
from pytest import mark
from copy import copy
import numpy as np


from myia.abstract import from_value
from myia.pipeline import standard_pipeline, standard_debug_pipeline
from myia.prim import ops as P
from myia.prim.py_implementations import \
    typeof, scalar_add, partial, bool_and
from myia.utils import no_prof, Profile
from myia.compile import load_backend

from .common import to_abstract_test


backend = load_backend(None)
if backend.__class__.__name__ == 'NumPyBackend':
    pytest.skip("can't test compile with the numpy backend")

compile_pipeline = backend.configure(standard_pipeline)

debug_fn = standard_debug_pipeline \
    .select('parse', 'resolve', 'infer', 'specialize', 'export')


def parse_compare(*tests, optimize=True, python=True, profile=no_prof):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    This uses the full myia pipeline.

    Arguments:
        tests: One or more inputs tuple.

    """
    pipeline = compile_pipeline if optimize else \
        compile_pipeline.configure({'opt.phases.main': []})

    def decorate(fn):
        def test(args):
            nonlocal profile
            if not isinstance(args, tuple):
                args = (args,)
            if python:
                ref_result = fn(*map(copy, args))
            argspec = tuple(from_value(arg, broaden=True) for arg in args)
            if profile is True:
                profile = Profile()
            res = pipeline.run(input=fn, argspec=argspec, profile=profile)
            profile.print()
            myia_fn = res['output']
            myia_result = myia_fn(*map(copy, args))
            if python:
                np.testing.assert_allclose(ref_result, myia_result)

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@parse_compare((2, 3))
def test_simple(x, y):
    return x + y


@parse_compare((42,))
def test_constant(x):
    return x == 42


@parse_compare((False, True), (True, True), (True, False), (False, False))
def test_bool_and(x, y):
    return bool_and(x, y)


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
    def fsum(x, a):
        if x == 1:
            return a
        else:
            return fsum(x - 1, a + x)
    return fsum(x, 1)


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
        return x * y

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, y) + choose(not c)(x, y)


@parse_compare((15, 17), optimize=False)
def test_partial_prim(x, y):
    return partial(scalar_add, x)(y)


def test_switch_nontail():
    def fn(x, y):
        def f1():
            return x

        def f2():
            return y

        a = P.switch(x > y, f1, f2)()
        return a * a

    i64 = typeof(1)
    argspec = (to_abstract_test(i64), to_abstract_test(i64))
    myia_fn = compile_pipeline.run(input=fn,
                                   argspec=argspec)['output']

    for test in [(6, 23, 23**2), (67, 23, 67**2)]:
        *args, expected = test
        assert myia_fn(*args) == expected


@parse_compare((None,),
               (42,))
def test_is_(x):
    return x is None


@parse_compare((None,),
               (42,))
def test_is_not(x):
    return x is not None
