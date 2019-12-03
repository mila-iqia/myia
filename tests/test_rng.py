"""Test myi public API."""

import numpy as np

from myia import myia
from myia.operations import random_initialize, random_uint32


def run_generator():
    rstate = random_initialize(12345678)
    r0, v0 = random_uint32(rstate, (2, 2))
    r1, v1 = random_uint32(r0, (2, 2))
    return r1, v0, v1


@myia(backend='pytorch')
def run_pytorch():
    return run_generator()


@myia(backend='relay')
def run_relay():
    return run_generator()


def run_random_generator(fn, expected_values):
    r, v0, v1 = fn()
    two_exp_32 = np.uint64(np.iinfo(np.uint32).max) + np.uint64(1)
    for v, expected in zip((v0, v1), expected_values):
        assert v.dtype == 'uint32'
        assert np.all(0 <= v)
        assert np.all(v < two_exp_32)
        assert np.all(v == expected)
    return r


def test_pytorch():
    expected = (
        np.asarray(
            [[3422054626, 1376353668], [825546192, 1797302575]],
            dtype='uint32'),
        np.asarray(
            [[1514647480, 548814914], [3847607334, 2603396401]],
            dtype='uint32')
    )
    run_random_generator(run_pytorch, expected)


def test_relay():
    expected = (
        np.asarray(
            [[29855489, 3396295364], [1662206086, 2440707850]],
            dtype='uint32'),
        np.asarray(
            [[3733719701, 3996474388], [316852167, 779101904]],
            dtype='uint32')
    )
    rstate = run_random_generator(run_relay, expected)
    key, counter = rstate
    assert key == 12345678
    assert counter == 2
