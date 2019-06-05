import pytest
import math
from copy import copy
import numpy as np

from myia.abstract import from_value
from myia.compile.backends import load_backend, LoadingError, UnknownBackend, \
    parse_default
from myia.pipeline import standard_pipeline
from myia.prim.py_implementations import distribute, scalar_to_array, dot, \
    scalar_add, array_reduce, transpose
from myia import dtype

from ..common import MA, MB


@pytest.fixture(params=[
    pytest.param(('nnvm', {'target': 'cpu', 'device_id': 0}), id='nnvm-cpu'),
    pytest.param(('relay', {'target': 'cpu', 'device_id': 0}), id='relay-cpu'),
    pytest.param(('pytorch', {'device': 'cpu'}), id='pytorch-cpu')])
def backend_opt(request):
    name, options = request.param
    return BackendOption(name, options)


class BackendOption:
    def __init__(self, backend, backend_options):
        try:
            load_backend(backend)
        except LoadingError as e:
            pytest.skip(f"Can't load {backend}: {e.__cause__}")
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
        def test(backend_opt, args):
            if not isinstance(args, tuple):
                args = (args,)
            ref_result = fn(*map(copy, args))
            argspec = tuple(from_value(arg, broaden=True) for arg in args)
            res = backend_opt.pip(input=fn, argspec=argspec)
            myia_fn = res['output']
            myia_args = backend_opt.convert_args(args)
            myia_result = myia_fn(*myia_args)
            np.testing.assert_allclose(ref_result, myia_result)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


def test_default_backend():
    import os

    before = os.environ.get('MYIA_BACKEND', None)
    try:
        os.environ['MYIA_BACKEND'] = 'pytorch'
        assert parse_default() == ('pytorch', {})

        os.environ['MYIA_BACKEND'] = 'pytorch?target=cpu'
        assert parse_default() == ('pytorch', {'target': 'cpu'})

        os.environ['MYIA_BACKEND'] = 'relay?target=cpu&device_id=0'
        assert parse_default() == ('relay', {'target': 'cpu',
                                             'device_id': '0'})
    finally:
        # Make sure we don't switch the default for other tests.
        if before is None:
            del os.environ['MYIA_BACKEND']
        else:
            os.environ['MYIA_BACKEND'] = before


def test_load_backend_unknown():
    with pytest.raises(UnknownBackend):
        load_backend('_fake_name_')


def test_backend_error():
    from myia.compile.backends import _backends, register_backend
    name = '__testing_name000_'

    def f():
        raise ValueError('test')

    register_backend(name, f)

    with pytest.raises(LoadingError):
        load_backend(name)

    del _backends[name]


def test_dlpack(backend_opt):
    backend = backend_opt.pip.steps.compile.backend
    v = MA(4, 3)
    nv = backend.from_numpy(v)
    dv = backend.to_dlpack(nv)
    nv2 = backend.from_dlpack(dv)
    v2 = backend.to_numpy(nv2)
    assert (v == v2).all()


def test_check_array_errors(backend_opt):
    backend = backend_opt.pip.steps.compile.backend
    with pytest.raises(Exception):
        backend.check_array(MA(1, 2), dtype.Float[64])

    bv = backend.from_numpy(MA(1, 2))
    with pytest.raises(Exception):
        backend.check_array(bv, dtype.Float[32])


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


@parse_compare((2,))
def test_distribute2(x):
    return distribute(scalar_to_array(x), (1,))


@parse_compare(np.ones((1, 3)), np.ones((3,)))
def test_distribute3(x):
    return distribute(x, (2, 3))


@parse_compare(MA(2, 3))
def test_distribute4(x):
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
def test_array_reduce3(x):
    return array_reduce(scalar_add, x, ())


@parse_compare((MA(2, 3),))
def test_transpose(x):
    return transpose(x, (1, 0))


@parse_compare((3, 4))
def test_make_tuple(a, b):
    return (a, b)


@parse_compare((True, 42, 33))
def test_call_hof(c, x, y):
    def f1(x):
        return x + y

    def f2(x):
        return x * y

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x) + choose(not c)(x)
