import pytest

from myia.pipeline import scalar_parse as parse, scalar_pipeline
from myia.parser import MyiaSyntaxError, MyiaDisconnectedCodeWarning


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


def test_no_return():
    def f():
        1 + 2
    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_no_return_while():
    def f():
        def g():
            y = 0
            x = 0
            while x < 10:
                x = x + 1
            while y < 10:
                y = y + 1
        return g()
    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_maybe():
    def f():  # pragma: no cover
        while True:
            x = 2
        return x
    with pytest.raises(Exception):
        # This shouldn't resolve
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

    # No resolve
    parse2 = scalar_pipeline \
        .select('parse') \
        .make_transformer('input', 'graph')

    parse2(g)

    def h():
        return 2 + 2


def test_unsupported_AST__error():
    def a0():
        _a0 = {}  # noqa: F841
    with pytest.raises(MyiaSyntaxError):
        parse(a0)

    def a1():
        pass
    with pytest.raises(MyiaSyntaxError):
        parse(a1)

    def a2():
        import builtins  # noqa: F401
    with pytest.raises(MyiaSyntaxError):
        parse(a2)

    def a3():
        assert False
    with pytest.raises(MyiaSyntaxError):
        parse(a3)


def test_disconnected_from_output__warning():
    def a0():
        print(1)
        return 1
    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a0)

    def a1():
        def b1():
            return 1
        b1()
        return 1
    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a1)

    # This tests that comments are not raised as warnings.
    def a2():
        """Comment: Blah Blah Blah
        """
        return 1
    with pytest.warns(None) as record:
        parse(a2)
    assert len(record) == 0

    def a3():
        x = 1
        return 1
    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a3)

    def a4():
        def b4():
            def c4():
                x = 1
                return 1
            return c4()
        return b4
    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a4)
