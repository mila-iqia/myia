from myia.compile.backends import parse_default


def test_default_backend():
    import os

    before = os.environ.get("MYIA_BACKEND", None)
    try:
        os.environ["MYIA_BACKEND"] = "relay?target=cpu&device_id=0"
        assert parse_default() == ("relay", {"target": "cpu", "device_id": "0"})
    finally:
        # Make sure we don't switch the default for other tests.
        if before is None:
            del os.environ["MYIA_BACKEND"]
        else:
            os.environ["MYIA_BACKEND"] = before
