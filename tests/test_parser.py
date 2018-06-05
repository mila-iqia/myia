import pytest

from myia.api import parse


def test_undefined():
    def f():  # pragma: no cover
        return c  # noqa
    with pytest.raises(NameError):
        parse(f)


def test_defined_later():
    def f():  # pragma: no cover
        return c  # noqa
    with pytest.raises(UnboundLocalError):
        parse(f)
    c = 3


def test_maybe():
    def f():  # pragma: no cover
        while True:
            x = 2
        return x
    with pytest.raises(Exception):
        # This shouldn't resolve
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


def test_forward_reference():
    def g():
        return h()

    parse(g, resolve_globals=False)

    def h():
        return 2 + 2
