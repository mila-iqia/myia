import pytest

from myia.api import parse


def test_undefined():
    def f():  # pragma: no cover
        return c  # noqa
    with pytest.raises(ValueError):
        parse(f)


@pytest.mark.xfail(reason='x is possibly undefined')
def test_maybe():
    def f():  # pragma: no cover
        while True:
            x = 2
        return x
    parse(f)


def test_unsupported():
    def f():  # pragma: no cover
        assert False
    with pytest.raises(NotImplementedError):
        parse(f)


def test_expression_statements():
    def f(x):  # pragma: no cover
        """Foo."""
        print(x)
        return x
    parse(f)


def _global_f():
    return 42


def test_global_nested():
    def g():
        def h():
            return _global_f()
        return h()

    parse(g)
