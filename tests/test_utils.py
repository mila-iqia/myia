from myia.utils import Named


def test_named():
    named = Named('foo')
    assert repr(named) == 'foo'
