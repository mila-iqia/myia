from myia.utils.misc import Named


def test_str():
    assert str(Named("TEST")) == "TEST"
