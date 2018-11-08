from copy import copy
from pytest import mark

from myia.api import standard_pipeline


debug_lin_pipeline = standard_pipeline.configure({
    'compile.linear_impl': 'debug'})


def parse_compare(*tests):
    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)
            py_result = fn(*map(copy, args))
            argspec = tuple({'value': a} for a in args)
            res = debug_lin_pipeline.run(input=fn, argspec=argspec)
            myia_fn = res['output']
            myia_result = myia_fn(*map(copy, args))
            assert py_result == myia_result

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@parse_compare((1,))
def test_debug_add(a):
    return a + 2


@parse_compare((2, 3))
def test_debug_floordiv(x, y):
    return x // y
