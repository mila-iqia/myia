import pytest
import math
from copy import copy
import numpy as np

from myia.abstract import from_value
from myia.compile.backends import load_backend, LoadingError
from myia.pipeline import standard_pipeline
from myia.prim.py_implementations import distribute, scalar_to_array, dot, \
    scalar_add, array_reduce, transpose

from ..common import MA, MB


@pytest.fixture(params=[('nnvm', {'target': 'cpu', 'device_id': 0})])
def backend(request):
    name, options = request.param
    return BackendOption(name, options)


class BackendOption:
    def __init__(self, backend, backend_options):
        try:
            load_backend(backend)
        except LoadingError as e:
            pytest.skip(f"Can't load {backend}: {e.message}")
        self.pip = standard_pipeline.configure({
            'compile.backend': backend,
            'compile.backend_options': backend_options
        }).make()

    def convert_args(self, args):
        return tuple(self.convert_arg(arg) for arg in args)

    def convert_arg(self, arg):
        if isinstance(arg, np.ndarray):
            return self.pip.steps.compile.backend.from_numpy(arg)
        elif isinstance(arg, tuple):
            return tuple(self.convert_arg(e) for e in arg)
        elif isinstance(arg, list):
            return list(self.convert_arg(e) for e in arg)
        elif isinstance(arg, (int, bool, float)):
            return arg
        # dataclasses
        else:
            raise ValueError(f"what is this: {arg}")


def parse_compare(*tests):
    """Decorate a function to run it against pure python.

    This will run and compare the function using all available backends.
    """
    def decorate(fn):
        def test(backend, args):
            if not isinstance(args, tuple):
                args = (args,)
            ref_result = fn(*map(copy, args))
            argspec = tuple(from_value(arg, broaden=True) for arg in args)
            res = backend.pip(input=fn, argspec=argspec)
            myia_fn = res['output']
            myia_args = backend.convert_args(args)
            myia_result = myia_fn(*myia_args)
            np.testing.assert_allclose(ref_result, myia_result)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@parse_compare((2, 3))
def test_add(x, y):
    return x + y


@parse_compare((2, 3))
def test_sub(x, y):
    return x - y


@parse_compare((2, 3))
def test_mul(x, y):
    return x * y


@pytest.mark.xfail(reason="scalar_cast is needed for ints")
@parse_compare((2, 3), (2.0, 3.0))
def test_truediv(x, y):
    return x / y


@parse_compare((2, 3), (2.0, 3.0))
def test_floordiv(x, y):
    return x // y


@parse_compare((2, 3))
def test_mod(x, y):
    return x % y


@parse_compare((2.0, 3.0))
def test_pow(x, y):
    return x ** y


@parse_compare((2,))
def test_uadd(x):
    return +x


@parse_compare((2,))
def test_usub(x):
    return -x


@parse_compare((2.0,))
def test_exp(x):
    return math.exp(x)


@parse_compare((2.0,))
def test_log(x):
    return math.log(x)


@pytest.mark.xfail(reason="not implemented")
@parse_compare((2.0,))
def test_tan(x):
    return math.tan(x)


@parse_compare((0.3,))
def test_tanh(x):
    return math.tanh(x)


@parse_compare((2, 3))
def test_eq(x, y):
    return x == y


@parse_compare((2, 3))
def test_lt(x, y):
    return x < y


@parse_compare((2, 3))
def test_gt(x, y):
    return x > y


@parse_compare((2, 3))
def test_ne(x, y):
    return x != y


@parse_compare((2, 3))
def test_le(x, y):
    return x <= y


@parse_compare((2, 3))
def test_ge(x, y):
    return x >= y


@parse_compare((True, False), (True, True))
def test_bool_eq(x, y):
    return x == y


@parse_compare((2,))
def test_to_array(x):
    return scalar_to_array(x)


@parse_compare((False,), (True,))
def test_bool_not(x,):
    return not x


@parse_compare((2,))
def test_distribute(x):
    return distribute(scalar_to_array(x), (2, 3))


@parse_compare(np.ones((1, 3)), np.ones((3,)))
def test_distribute2(x):
    return distribute(x, (2, 3))


@parse_compare(MA(2, 3))
def test_distribute3(x):
    return distribute(x, (2, 3))


@parse_compare((MA(2, 3), MB(3, 4)))
def test_dot(x, y):
    return dot(x, y)


@parse_compare((MA(2, 3), MB(2, 3)),
               (MA(1, 3), MB(2, 3)),
               (MA(2, 1), MB(2, 3)))
def test_array_map(x, y):
    return x + y


@parse_compare((MA(2, 3),), (MA(1, 3),))
def test_array_reduce(x):
    return array_reduce(scalar_add, x, (1, 3))


@parse_compare((MA(2, 3),))
def test_array_reduce2(x):
    return array_reduce(scalar_add, x, (3,))


@parse_compare((MA(2, 3),))
def test_transpose(x):
    return transpose(x, (1, 0))
