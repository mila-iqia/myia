from myia.compile.backends import parse_default


def test_default_backend():
    import os

    before = os.environ.get("MYIA_BACKEND", None)
    try:
        os.environ["MYIA_BACKEND"] = "pytorch"
        assert parse_default() == ("pytorch", {})

        os.environ["MYIA_BACKEND"] = "pytorch?target=cpu"
        assert parse_default() == ("pytorch", {"target": "cpu"})
    finally:
        # Make sure we don't switch the default for other tests.
        if before is None:
            del os.environ["MYIA_BACKEND"]
        else:
            os.environ["MYIA_BACKEND"] = before
