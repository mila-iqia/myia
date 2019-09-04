# Most of the tests ar in test_backend, this is just for pytorch-specific
# tests that can't be made generic.

import pytest

try:
    from myia.compile.backends import pytorch
except ImportError:
    pytestmark = pytest.mark.skip(f"Can't import pytorch")


def test_pytorch_type_convert():
    with pytest.raises(TypeError):
        pytorch.type_to_pytorch_type(object())
