# Most of the tests ar in test_backend, this is just for relay-specific
# tests that can't be made generic.

import pytest

try:
    from myia.compile.backends import relay
except ImportError:
    pytestmark = pytest.mark.skip(f"Can't import relay")


def test_relay_type_convert():
    with pytest.raises(ValueError):
        relay.to_relay_type(object())


@pytest.mark.gpu
def test_relay_backend_bad_device():
    with pytest.raises(RuntimeError):
        relay.RelayBackend(target='cuda', device_id=31)
