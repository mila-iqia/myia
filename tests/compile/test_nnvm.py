# Most of the tests ar in test_backend, this is just for nnvm-specific
# tests that can't be made generic.

import pytest

from myia.compile import load_backend


@pytest.mark.gpu
def test_nnvm_backend_bad_device():
    with pytest.raises(Exception):
        load_backend('nnvm', dict(target="cuda", device_id=31))
