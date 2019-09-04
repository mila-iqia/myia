# Most of the tests ar in test_backend, this is just for nnvm-specific
# tests that can't be made generic.

import pytest

try:
    from myia.compile.backends import nnvm
except ImportError:
    pytestmark = pytest.mark.skip(f"Can't import nnvm")


@pytest.mark.gpu
def test_nnvm_backend_bad_device():
    with pytest.raises(RuntimeError):
        nnvm.NNVMBackend(target='cuda', device_id=31)
