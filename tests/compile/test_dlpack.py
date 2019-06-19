import pytest

from myia.compile.backends import load_backend, LoadingError
import numpy as np


def safe_load(name):
    try:
        return load_backend(name)
    except LoadingError:
        return None


@pytest.mark.skip('Segfaults with pytorch 1.0.1')
def test_dlpack_cross():
    relay = safe_load('relay')
    nnvm = safe_load('nnvm')
    pytorch = safe_load('pytorch')

    backends = [b for b in [relay, nnvm, pytorch] if b is not None]

    for src in backends:
        for dst in backends:
            print(f"dlpack: {src} -> {dst}")
            a_np = np.array([1, 2, 3])
            a_src = src.from_numpy(a_np)
            dl_src = src.to_dlpack(a_src)
            a_dst = dst.from_dlpack(dl_src)
            b_np = dst.to_numpy(a_dst)
            if not (a_np == b_np).all():
                AssertionError(f"Error in dlpack transfer from {src} to {dst}")
