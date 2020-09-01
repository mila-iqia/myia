import pytest

from myia.compile.backends import LoadingError, UnknownBackend, load_backend
from myia.operations import (
    array_reduce,
    array_to_scalar,
    reshape,
    scalar_add,
    scalar_to_array,
)
from myia.testing.common import AN, MA
from myia.testing.multitest import mt, run, run_gpu


def test_load_backend_unknown():
    with pytest.raises(UnknownBackend):
        load_backend("_fake_name_")


def test_backend_error():
    from myia.compile.backends import _backends, register_backend

    name = "__testing_name000_"

    def format():
        return {}

    def f():
        raise ValueError("test")

    register_backend(name, f, format)

    with pytest.raises(LoadingError):
        load_backend(name)

    del _backends[name]


@run(MA(2, 3))
def test_reshape2(x):
    return reshape(x, (6,))


@mt(run(MA(2, 3)), run(MA(1, 3)))
def test_array_reduce(x):
    return array_reduce(scalar_add, x, (1, 3))


@run(MA(2, 3))
def test_array_reduce2(x):
    return array_reduce(scalar_add, x, (3,))


@run_gpu(MA(1, 1))
def test_array_to_scalar(x):
    return array_to_scalar(reshape(x, ()))


@mt(run(2, 3), run(2.0, 3.0))
def test_truediv(x, y):
    return x / y


@run_gpu(2)
def test_to_array(x):
    return scalar_to_array(x, AN)


@mt(run(None), run(True), run(False))
def test_bool_and_nil_args(x):
    return x


@run(3)
def test_return_tuple(x):
    return (1, 2, x)


@pytest.mark.xfail  # MyiaTypeError: AbstractTuple vs AbstractTaggedUnion
@run(())
def test_return_list():
    return [1, 2, 3]


ll = [1, 2, 3]


@pytest.mark.xfail  # unhashable type in cse
@run(())
def test_constant_list():
    return ll


a = MA(2, 3)


@pytest.mark.xfail  # unhashable type in cse
@run(())
def test_constant_array():
    return a
