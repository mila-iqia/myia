import pytest

from myia.api import parse


def test_simple():
    def f(x, y):
        z = x + y
        return z
    parse(f)


def test_anon_expression():
    def f(x, y):
        return x + y
    parse(f)


def test_globals():
    c = 2

    def f(x):
        return x + c
    parse(f)


def test_calls():
    def f():
        return 2

    def g():
        return f()
    parse(g)


def test_unsupported_object():
    c = object()

    def f():
        return c
    with pytest.raises(ValueError):
        parse(f)


def test_conditional():
    def f():
        if True:
            x = 2
        else:
            x = 1
        return x
    parse(f)


def test_empty_conditional():
    def f(x):
        if True:
            x = 2
        return x
    parse(f)


def test_multiple_return():
    def f(x):
        if True:
            return 1
        return 2
    parse(f)


def test_undefined():
    def f():
        return c  # noqa
    with pytest.raises(ValueError):
        parse(f)


def test_nested():
    def f():
        x = 2

        def g():
            return x
        return g
    parse(f)


def test_unpacking():
    def f():
        x, y = f()
    with pytest.raises(NotImplementedError):
        parse(f)


def test_while():
    def f():
        x = 1
        while x:
            x = x + 1
        return x
    parse(f)


@pytest.mark.xfail(reason='x is possibly undefined')
def test_maybe():
    def f():
        while True:
            x = 2
        return x
    parse(f)


def test_unsupported():
    def f():
        assert False
    with pytest.raises(NotImplementedError):
        parse(f)


def test_expression_statements():
    def f(x):
        """Foo."""
        print(x)
        return x
    parse(f)
