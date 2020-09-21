from typing import Dict, List, Tuple

import numpy as np
import pytest

from myia import myia
from myia.compile.backends import get_backend_names
from myia.operations import random_initialize, random_uint32
from myia.utils import AnnotationMismatchError
from myia.utils.misc import RandomStateWrapper


@pytest.fixture(
    params=[pytest.param(backend) for backend in get_backend_names()]
)
def _backend_fixture(request):
    return request.param


def test_scalar(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f(x: int, y: float) -> np.float32:
        return np.float32(np.float64(x) * np.float64(y))

    @myia(backend=backend)
    def g(a, b) -> np.float32:
        return a * b

    @myia(backend=backend)
    def h(a, b):
        c: float = a * b
        return 2 * c

    assert f(2, 4.5) == np.float32(9)
    assert g(np.float32(2), np.float32(3)) == np.float32(6)
    assert h(1.0, 2.0) == 4.0

    with pytest.raises(AnnotationMismatchError):
        # wrong type for first argument
        f(2.0, 4.5)

    with pytest.raises(AnnotationMismatchError):
        # wrong type for 2nd argument
        f(2, 4)

    with pytest.raises(AnnotationMismatchError):
        # wrong output type
        g(np.arange(1), np.arange(1))

    with pytest.raises(AnnotationMismatchError):
        # wrong output scalar type
        g(2, 3)

    with pytest.raises(AnnotationMismatchError):
        # Wrong internal variable type
        h(1, 2)


def test_tuple(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f(x: tuple):
        return x[0] + x[1]

    @myia(backend=backend)
    def g(x: Tuple) -> tuple:
        # to check if `Tuple` is parsed correctly as `tuple`.
        return x

    @myia(backend=backend)
    def h(x: Tuple[float, int]):
        return x[0] + float(x[1])

    @myia(backend=backend)
    def j(x):
        y: tuple = x
        return y[0]

    assert f((2, 3)) == 5
    assert g((2,)) == (2,)
    assert h((2.0, 3)) == 5.0
    assert j((7, 5)) == 7

    with pytest.raises(AnnotationMismatchError):
        # wrong argument type
        f([2, 3])

    with pytest.raises(AnnotationMismatchError):
        # wrong argument type
        g([2, 3])

    with pytest.raises(AnnotationMismatchError):
        # wrong tuple elements type
        h((2.0, 3.0))

    with pytest.raises(AnnotationMismatchError):
        # wrong tuple length
        h((1.0, 2, 3))

    with pytest.raises(AnnotationMismatchError):
        # wrong internal type
        j(7)


def test_list(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f(x: list):
        return x[0] + 2

    @myia(backend=backend)
    def g(x: List[np.int16]):
        return x[0] + 2

    @myia(backend=backend)
    def h(x):
        y: list = x
        return y[0] + 2

    assert f([5, 3]) == 7

    assert g([np.int16(10), np.int16(3)]) == 12

    assert h([5, 3]) == 7

    with pytest.raises(AnnotationMismatchError):
        # wrong argument type
        f((5, 3))

    with pytest.raises(AnnotationMismatchError):
        # wrong list element type
        g([5, 3])

    with pytest.raises(AnnotationMismatchError):
        h((5, 3))


def test_dict(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f(x: Dict[str, np.float32]):
        return np.float32(x["value"]) * np.float32(2.5)

    @myia(backend=backend)
    def g(x: dict):
        return x

    @myia(backend=backend)
    def h(x: Dict[Tuple[int, int], int]):
        return x

    @myia(backend=backend)
    def j(x: Dict[int, int]):
        return x

    @myia(backend=backend)
    def k(x):
        y: Dict[str, np.float32] = x
        return y["test"]

    d1 = {"test": 5, "value": 11}
    d2 = {"test": np.float32(5), "value": np.float32(11)}

    assert f(d2) == 27.5
    assert k(d2) == np.float32(5)

    with pytest.raises(AnnotationMismatchError):
        # wrong dict value type
        f(d1)

    with pytest.raises(AnnotationMismatchError):
        # wrong argument type
        g((1, 2))

    with pytest.raises(AnnotationMismatchError):
        # unsupported dict key type
        h(d1)

    with pytest.raises(AnnotationMismatchError):
        # wrong dict key type
        j(d1)

    with pytest.raises(AnnotationMismatchError):
        # wrong internal type
        k(d1)


def test_ndarray(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f(a, b: np.ndarray) -> np.ndarray:
        return a * b

    @myia(backend=backend)
    def g(a):
        x: np.ndarray = 2 * a + 1
        return x[0, 0].item()

    arr = np.ones((2, 2), dtype="int64")

    assert np.all(f(2, arr) == 2 * arr)
    assert g(arr) == 3

    with pytest.raises(AnnotationMismatchError):
        # wrong type for 2nd argument
        f(2, 2)

    with pytest.raises(AnnotationMismatchError):
        # wrong internal type
        g(0)


def test_random_state_wrapper(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def f() -> RandomStateWrapper:
        rstate: RandomStateWrapper = random_initialize(10)
        r0, _ = random_uint32(rstate, ())
        return r0

    @myia(backend=backend)
    def g(rstate: RandomStateWrapper):
        return rstate

    g(f())

    with pytest.raises(AnnotationMismatchError):
        # wrong argument type
        g(0)
