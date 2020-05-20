"""Test myi public API."""

import numpy as np
import pytest

from myia import myia
from myia.operations import random_initialize, random_uint32


@pytest.fixture(params=[pytest.param("pytorch"), pytest.param("relay")])
def _backend_fixture(request):
    return request.param


EXPECTED = {
    "pytorch": (
        np.asarray(
            [[3422054626, 1376353668], [825546192, 1797302575]], dtype="uint32"
        ),
        np.asarray(
            [[1514647480, 548814914], [3847607334, 2603396401]], dtype="uint32"
        ),
        np.asarray([1379542846], dtype="uint32"),
        1617509384,
    ),
    "relay": (
        np.asarray(
            [[29855489, 3396295364], [1662206086, 2440707850]], dtype="uint32"
        ),
        np.asarray(
            [[3733719701, 3996474388], [316852167, 779101904]], dtype="uint32"
        ),
        np.asarray([3997522715], dtype="uint32"),
        2591148269,
    ),
}
TWO_EXP_32 = np.uint64(np.iinfo(np.uint32).max) + np.uint64(1)


def _test_output(backend, final_rstate, *generated_values):
    expected_values = EXPECTED[backend]
    assert len(expected_values) == len(generated_values)
    for v, expected in zip(generated_values, expected_values):
        assert v.dtype == "uint32"
        assert np.all(0 <= v)
        assert np.all(v < TWO_EXP_32)
        assert np.all(v == expected)
    if backend == "relay":
        key, counter = final_rstate.state
        assert key == 12345678
        assert counter == 4


def test_init_random_combined(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def fn():
        rstate = random_initialize(12345678)
        r0, v0 = random_uint32(rstate, (2, 2))
        r1, v1 = random_uint32(r0, (2, 2))
        r2, v2 = random_uint32(r1, (1,))
        r3, v3 = random_uint32(r2, ())
        return r3, v0, v1, v2, v3

    _test_output(backend, *fn())


def test_init_random_separated(_backend_fixture):
    backend = _backend_fixture

    @myia(backend=backend)
    def init():
        return random_initialize(12345678)

    @myia(backend=backend)
    def gen_2_2(rng):
        return random_uint32(rng, (2, 2))

    @myia(backend=backend)
    def gen_1(rng):
        return random_uint32(rng, (1,))

    @myia(backend=backend)
    def gen_scalar(rng):
        return random_uint32(rng, ())

    rstate = init()
    r0, v0 = gen_2_2(rstate)
    r1, v1 = gen_2_2(r0)
    r2, v2 = gen_1(r1)
    r3, v3 = gen_scalar(r2)

    _test_output(backend, r3, v0, v1, v2, v3)
