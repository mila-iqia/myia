# Most of the tests ar in test_backend, this is just for pytorch-specific
# tests that can't be made generic.

import numpy as np
import pytest
from myia import dtype

try:
    from myia.compile.backends import pytorch
except ImportError:
    pytestmark = pytest.mark.skip(f"Can't import pytorch")


def test_pytorch_type_convert():
    with pytest.raises(TypeError):
        pytorch.type_to_pytorch_type(object())


@pytest.mark.gpu
def test_pytorch_check_array():
    backend_cuda = pytorch.PyTorchBackend(device='cuda:0')
    backend_cpu = pytorch.PyTorchBackend(device='cpu')

    v = np.ndarray([1, 2, 3])
    tp = dtype.Int[64]

    t_cuda = backend_cuda.from_numpy(v)
    t_cpu = backend_cpu.from_numpy(v)

    with pytest.raises(RuntimeError):
        backend_cuda.check_array(t_cpu, tp)

    with pytest.raises(RuntimeError):
        backend_cpu.check_array(t_cuda, tp)
