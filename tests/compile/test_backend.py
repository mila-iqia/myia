import math

import numpy as np
import pytest

from myia.abstract import from_value
from myia.api import to_device
from myia.compile.backends import (
    LoadingError,
    UnknownBackend,
    load_backend,
    parse_default,
)
from myia.operations import (
    array_reduce,
    distribute,
    dot,
    reshape,
    scalar_add,
    scalar_to_array,
    transpose,
)
from myia.pipeline import standard_pipeline

from ..common import AN, MA, MB, to_abstract_test
from ..multitest import Multiple, mt, myia_function_test


class BackendOption:
    def __init__(self, backend, backend_options):
        try:
            self.backend = load_backend(backend, backend_options)
        except LoadingError as e:
            pytest.skip(f"Can't load {backend}: {e.__cause__}")
        self.pip = standard_pipeline.configure({
            'resources.backend.name': backend,
            'resources.backend.options': backend_options
        }).make()

    def convert_args(self, args):
        return tuple(to_device(arg, self.backend) for arg in args)


@myia_function_test(id='run_backend')
def _run_backend(self, fn, args, result=None, abstract=None,
                 backend=None):
    backend = BackendOption(*backend)

    if abstract is None:
        argspec = tuple(from_value(arg, broaden=True)
                        for arg in args)
    else:
        argspec = tuple(to_abstract_test(a) for a in abstract)

    def out():
        mfn = backend.pip(input=fn, argspec=argspec)
        myia_args = backend.convert_args(args)
        rval = mfn['output'](*myia_args)
        return rval

    if result is None:
        result = fn(*args)

    self.check(out, result)


run_backend = _run_backend.configure(
    backend=Multiple(
        pytest.param(('relay', {'target': 'cpu', 'device_id': 0}),
                     id='relay-cpu',
                     marks=pytest.mark.relay),
        pytest.param(('relay', {'target': 'cuda', 'device_id': 0}),
                     id='relay-cuda',
                     marks=[pytest.mark.relay, pytest.mark.gpu]),
        pytest.param(('pytorch', {'device': 'cpu'}),
                     id='pytorch-cpu',
                     marks=pytest.mark.pytorch),
        pytest.param(('pytorch', {'device': 'cuda'}),
                     id='pytorch-cuda',
                     marks=[pytest.mark.pytorch, pytest.mark.gpu])
    )
)


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

    def format():
        return {}

    def f():
        raise ValueError('test')

    register_backend(name, f, format)

    with pytest.raises(LoadingError):
        load_backend(name)

    del _backends[name]


@run_backend(2, 3)
def test_add(x, y):
    return x + y


@run_backend(2, 3)
def test_sub(x, y):
    return x - y


@run_backend(2, 3)
def test_mul(x, y):
    return x * y


@pytest.mark.xfail(reason="scalar_cast is needed for ints")
@mt(
    run_backend(2, 3),
    run_backend(2.0, 3.0),
)
def test_truediv(x, y):
    return x / y


@mt(
    run_backend(2, 3),
    run_backend(2.0, 3.0)
)
def test_floordiv(x, y):
    return x // y


@run_backend(2, 3)
def test_mod(x, y):
    return x % y


@run_backend(2.0, 3.0)
def test_pow(x, y):
    return x ** y


@run_backend(2)
def test_uadd(x):
    return +x


@run_backend(2)
def test_usub(x):
    return -x


@run_backend(2.0)
def test_exp(x):
    return math.exp(x)


@run_backend(2.0)
def test_log(x):
    return math.log(x)


@pytest.mark.xfail(reason="not implemented")
@run_backend(2.0)
def test_tan(x):
    return math.tan(x)


@run_backend(0.3)
def test_tanh(x):
    return math.tanh(x)


@run_backend(2, 3)
def test_eq(x, y):
    return x == y


@run_backend(2, 3)
def test_lt(x, y):
    return x < y


@run_backend(2, 3)
def test_gt(x, y):
    return x > y


@run_backend(2, 3)
def test_ne(x, y):
    return x != y


@run_backend(2, 3)
def test_le(x, y):
    return x <= y


@run_backend(2, 3)
def test_ge(x, y):
    return x >= y


@mt(
    run_backend(True, False),
    run_backend(True, True)
)
def test_bool_eq(x, y):
    return x == y


@run_backend(2)
def test_to_array(x):
    return scalar_to_array(x, AN)


@mt(
    run_backend(False),
    run_backend(True),
)
def test_bool_not(x,):
    return not x


@run_backend(2)
def test_distribute(x):
    return distribute(scalar_to_array(x, AN), (2, 3))


@run_backend(2)
def test_distribute2(x):
    return distribute(scalar_to_array(x, AN), (1,))


@mt(
    run_backend(np.ones((1, 3))),
    run_backend(np.ones((3,))),
)
def test_distribute3(x):
    return distribute(x, (2, 3))


@run_backend(MA(2, 3))
def test_distribute4(x):
    return distribute(x, (2, 3))


@run_backend(MA(2, 3))
def test_reshape(x):
    return reshape(x, (1, 3, 2, 1))


@run_backend(MA(2, 3))
def test_reshape2(x):
    return reshape(x, (6,))


@run_backend(MA(1, 3))
def test_reshape3(x):
    return reshape(x, (3,))


@run_backend(np.ones((1,)))
def test_reshape4(x):
    return reshape(x, ())


@run_backend(MA(2, 3), MB(3, 4))
def test_dot(x, y):
    return dot(x, y)


@mt(
    run_backend(MA(2, 3), MB(2, 3)),
    run_backend(MA(1, 3), MB(2, 3)),
    run_backend(MA(2, 1), MB(2, 3)),
)
def test_array_map(x, y):
    return x + y


@mt(
    run_backend(MA(2, 3)),
    run_backend(MA(1, 3)),
)
def test_array_reduce(x):
    return array_reduce(scalar_add, x, (1, 3))


@run_backend(MA(2, 3))
def test_array_reduce2(x):
    return array_reduce(scalar_add, x, (3,))


@run_backend(MA(2, 3))
def test_array_reduce3(x):
    return array_reduce(scalar_add, x, ())


@run_backend(MA(2, 3))
def test_transpose(x):
    return transpose(x, (1, 0))


@run_backend(3, 4)
def test_make_tuple(a, b):
    return (a, b)


@run_backend(True, 42, 33)
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


@mt(
    run_backend(None),
    run_backend(True),
    run_backend(False)
)
def test_bool_and_nil_args(x):
    return x


@run_backend(None)
def test_True_assign(_x):
    x = True
    return x


@run_backend(None)
def test_False_assign(_x):
    x = False
    return x


@run_backend(np.array(2))
def test_array_to_scalar(x):
    return x.item()
