# Most of the tests ar in test_backend, this is just for relay-specific
# tests that can't be made generic.

import pytest

from myia.compile import load_backend


@pytest.mark.relay
@pytest.mark.gpu
def test_relay_backend_bad_device():
    with pytest.raises(Exception):
        load_backend("relay", dict(target="cuda", device_id=31))


@pytest.mark.relay
def test_relay_backend_bad_exec():
    with pytest.raises(Exception):
        load_backend("relay", dict(target="cpu", exec_kind="potato"))
