from typing import Dict, List, Tuple

import numpy as np
import pytest

from myia import myia
from myia.operations import random_initialize, random_uint32
from myia.pipeline.annotation_validation import AnnotationValidationError
from myia.utils.misc import RandomStateWrapper


def test_scalar():
    @myia
    def f(x: int, y: float) -> np.float32:
        return np.float32(np.float64(x) * np.float64(y))

    @myia
    def g(a, b) -> np.float32:
        return a * b

    assert f(2, 4.5) == np.float32(9)

    assert g(np.float32(2), np.float32(3)) == np.float32(6)

    with pytest.raises(AnnotationValidationError):
        # wrong type for first argument
        f(2.0, 4.5)

    with pytest.raises(AnnotationValidationError):
        # wrong type for 2nd argument
        f(2, 4)

    with pytest.raises(AnnotationValidationError):
        # wrong output type
        g(np.arange(1), np.arange(1))

    with pytest.raises(AnnotationValidationError):
        # wrong output scalar type
        g(2, 3)


def test_tuple():
    @myia
    def f(x: tuple):
        return x[0] + x[1]

    @myia
    def f_(x: Tuple):
        # to check if `Tuple` is parsed correctly as `tuple`.
        return x

    @myia
    def g(x: Tuple[float, int]):
        return x[0] + float(x[1])

    assert f((2, 3)) == 5
    assert f_((2,)) == (2,)
    assert g((2.0, 3)) == 5.0

    with pytest.raises(AnnotationValidationError):
        # wrong argument type
        f([2, 3])

    with pytest.raises(AnnotationValidationError):
        f_([2, 3])

    with pytest.raises(AnnotationValidationError):
        # wrong tuple elements type
        g((2.0, 3.0))

    with pytest.raises(AnnotationValidationError):
        # wrong tuple length
        g((1.0, 2, 3))


def test_list():
    @myia
    def f(x: list):
        return x[0] + 2

    @myia
    def g(x: List[np.int16]):
        return x[0] + 2

    assert f([5, 3]) == 7

    assert g([np.int16(10), np.int16(3)]) == 12

    with pytest.raises(AnnotationValidationError):
        # wrong argument type
        f((5, 3))

    with pytest.raises(AnnotationValidationError):
        # wrong list element type
        g([5, 3])


def test_dict():
    @myia
    def f(x: Dict[str, np.float32]):
        return np.float32(x["value"]) * np.float32(2.5)

    @myia
    def g(x: Dict[str, int]):
        return x

    @myia
    def h(x: Dict[Tuple[int, int], int]):
        return x

    @myia
    def j(x: Dict[int, int]):
        return x

    d1 = {"test": 5, "value": 11}
    d2 = {"test": np.float32(5), "value": np.float32(11)}

    assert f(d2) == 27.5

    with pytest.raises(AnnotationValidationError):
        # wrong dict value type
        f(d1)

    with pytest.raises(AnnotationValidationError):
        # wrong argument type
        g((1, 2))

    with pytest.raises(AnnotationValidationError):
        # unsupported dict key type
        h(d1)

    with pytest.raises(AnnotationValidationError):
        # wrong dict key type
        j(d1)


def test_ndarray():
    @myia
    def f(a, b: np.ndarray) -> np.ndarray:
        return a * b

    arr = np.ones((2, 2), dtype="int64")

    assert np.all(f(2, arr) == 2 * arr)

    with pytest.raises(AnnotationValidationError):
        # wrong type for 2nd argument
        f(2, 2)


def test_random_state_wrapper():
    @myia
    def f(seed) -> RandomStateWrapper:
        rstate: RandomStateWrapper = random_initialize(np.uint32(seed))
        r0, _ = random_uint32(rstate, ())
        return r0

    @myia
    def g(rstate: RandomStateWrapper):
        return rstate

    g(f(10))

    with pytest.raises(AnnotationValidationError):
        # wrong argument type
        g(0)
