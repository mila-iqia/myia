"""Test RNG for Python backend."""

import numpy as np

from myia import myia
from myia.operations import random_initialize, random_uint32

EXPECTED = (
    np.asarray(
        [[1055721139, 3422054626], [2561641375, 1376353668]], dtype="uint32"
    ),
    np.asarray(
        [[1540998321, 825546192], [1627406507, 1797302575]], dtype="uint32"
    ),
    np.asarray([105017825], dtype="uint32"),
    1514647480,
)
TWO_EXP_32 = np.uint64(np.iinfo(np.uint32).max) + np.uint64(1)


def _test_output(_, *generated_values):
    expected_values = EXPECTED
    assert len(expected_values) == len(generated_values)
    for v, expected in zip(generated_values, expected_values):
        assert v.dtype == "uint32"
        assert np.all(0 <= v)
        assert np.all(v < TWO_EXP_32)
        assert np.all(v == expected)


def test_init_random_combined():
    backend = "python"

    @myia(backend=backend)
    def fn():
        rstate = random_initialize(12345678)
        r0, v0 = random_uint32(rstate, (2, 2))
        r1, v1 = random_uint32(r0, (2, 2))
        r2, v2 = random_uint32(r1, (1,))
        r3, v3 = random_uint32(r2, ())
        return r3, v0, v1, v2, v3

    _test_output(*fn())


def test_init_random_separated():
    backend = "python"

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

    _test_output(r3, v0, v1, v2, v3)
