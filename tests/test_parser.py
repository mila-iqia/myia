import re
import sys
import warnings
from typing import List

import numpy as np
import pytest

from myia.debug.traceback import myia_warning
from myia.ir import manage
from myia.parser import (
    MyiaDisconnectedCodeWarning,
    MyiaSyntaxError,
    parse as raw_parse,
)
from myia.pipeline import scalar_parse as parse, scalar_pipeline


def test_undefined():
    def f():
        return c  # noqa

    with pytest.raises(NameError):
        parse(f)


def test_defined_later():
    def f():
        return c  # noqa

    with pytest.raises(UnboundLocalError):
        parse(f)
    c = 3


def test_no_return():
    def f():
        """Hello, there is nothing here!"""

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
    def f():
        while True:
            x = 2
        return x

    with pytest.raises(Exception):
        # This shouldn't resolve
        parse(f)


def test_unsupported():
    def f():
        assert False

    with pytest.raises(MyiaSyntaxError):
        parse(f)


def test_expression_statements():
    def f(x):
        """Foo."""
        print(x)
        return x

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(f)


def _global_f():
    return 42


def test_global_nested():
    def g():
        def h():
            return _global_f()

        return h()

    parse(g)


def test_forward_reference_in_closure():
    def g(x):
        def h():
            return a * a

        a = x * x
        return h()

    parse(g)


def test_modifying_forward_reference():
    def g(x):
        def h():
            return a * a

        a = x * x
        b = h()
        a = a * a
        return b, h()

    with pytest.raises(MyiaSyntaxError):
        parse(g)


def test_mutual_recursion():
    def g(x):
        def rec1(y):
            return rec2(y - 1)

        def rec2(y):
            return rec1(y - 1)

        return rec1(x)

    parse(g)


def test_forward_reference():
    def g():
        return h()

    # No resolve
    parse2 = scalar_pipeline.select("resources", "parse").make_transformer(
        "input", "graph"
    )

    parse2(g)

    def h():
        return 2 + 2


def test_dict():
    def bad(x):
        return {x: 2}

    with pytest.raises(MyiaSyntaxError):
        parse(bad)


def test_parametric():
    def f(x, y=6):
        return x + y

    def g(x, y, *args):
        return x + args[0]

    def h(*args):
        return f(*args) * g(*args)

    def i(x, *, y):
        return x + y

    def j(**kwargs):
        return kwargs

    assert raw_parse(f).defaults == ["y"]
    assert raw_parse(g).vararg == "args"
    assert raw_parse(h).vararg == "args"
    assert raw_parse(i).kwonly == 1
    assert raw_parse(j).kwarg == "kwargs"


def test_annotation_parsing_typing():

    # Type annotation for b is wrong, but we use is here just for testing.
    def f(a: int, b: List[int]) -> bool:
        c: tuple = (2, 3)
        d: int = int(b + 1.5)
        return bool(a * b) * c[0] + d

    graph = raw_parse(f)
    manager = manage(graph)

    # Check parameters annotation.
    parameters = {p.debug.debug_name: p for p in graph.parameters}
    assert parameters["a"].annotation is int
    assert parameters["b"].annotation is not List
    assert parameters["b"].annotation is List[int]

    # Check return annotation.
    assert graph.return_.annotation is bool

    # Check variable annotations.
    variables_checked = 0
    for node in manager.all_nodes:
        name = node.debug.debug_name
        if name == "c":
            assert node.annotation is tuple
            variables_checked += 1
        elif name == "d":
            assert node.annotation is int
            variables_checked += 1
    assert variables_checked == 2


def test_annotation_parsing_numpy():
    def f(a: np.ndarray) -> np.ndarray:
        b: np.ndarray = np.arange(10)
        return a.sum() + b.sum()

    graph = raw_parse(f)
    manager = manage(graph)

    # Check parameters annotation.
    parameters = {p.debug.debug_name: p for p in graph.parameters}
    assert parameters["a"].annotation is np.ndarray

    # Check return annotation.
    assert graph.return_.annotation is np.ndarray

    # Check variable annotations.
    variables_checked = 0
    for node in manager.all_nodes:
        name = node.debug.debug_name
        if name == "b":
            assert node.annotation is np.ndarray
            variables_checked += 1
    assert variables_checked == 1


def test_fn_param_same_name():
    def a(a):
        return a + 1

    fa = raw_parse(a)
    assert fa.output.inputs[1] is fa.parameters[0]


def test_unsupported_AST__error():
    def a1():
        import builtins  # noqa: F401

        return None

    with pytest.raises(MyiaSyntaxError):
        parse(a1)


def test_disconnected_from_output__warning():
    def a0():
        print(1)
        return 1

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a0)

    def a1():
        x = 1
        while x < 5:
            x = x + 1
        print(1)
        return 1

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a1)

    def a2():
        def b2():
            return 1

        b2()
        return 1

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a2)

    # This tests that comments are not raised as warnings.
    def a3():
        """Comment: Blah Blah Blah
        """
        return 1

    with pytest.warns(None) as record:
        parse(a3)
    assert len(record) == 0

    def a4():
        x = 1  # noqa: F841
        return 1

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a4)

    def a5():
        def b5():
            def c5():
                x = 1  # noqa: F841
                return 1

            return c5()

        return b5

    with pytest.warns(MyiaDisconnectedCodeWarning):
        parse(a5)


def test_no_return__format(capsys):
    def f():
        pass

    try:
        parse(f)
    except MyiaSyntaxError:
        sys.excepthook(*sys.exc_info())

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): def f\(\):\n"
        + r"(.+?)  \^\^\^\^\^\^\^\^\n"
        + r"(.+?):     pass\n"
        + r"(.+?)  \^\^\^\^\^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaSyntaxError: Function doesn't return a value"
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None


def test_no_return_while__format(capsys):
    def f():
        def g():
            y = 0
            x = 0
            while x < 10:
                x = x + 1
            while y < 10:
                y = y + 1

        return g()

    try:
        parse(f)
    except MyiaSyntaxError:
        sys.excepthook(*sys.exc_info())

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): def g\(\):\n"
        + r"     \^\^\^\^\^\^\^\^\n"
        + r"(.+?):     y = 0\n"
        + r"     \^\^\^\^\^\^\^\^\^\n"
        + r"(.+?):     x = 0\n"
        + r"     \^\^\^\^\^\^\^\^\^\n"
        + r"(.+?):     while x < 10:\n"
        + r"     \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\n"
        + r"(.+?):         x = x \+ 1\n"
        + r"     \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\n"
        + r"(.+?):     while y < 10:\n"
        + r"     \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\n"
        + r"(.+?):         y = y \+ 1\n"
        + r"     \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaSyntaxError: Function doesn't return a value in all cases"
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None


def test_unsupported_AST__error__format(capsys):
    def a1():
        import builtins  # noqa: F401

        return 1

    try:
        parse(a1)
    except MyiaSyntaxError:
        sys.excepthook(*sys.exc_info())

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): import builtins  # noqa: F401\n"
        + r"(.+?)  \^\^\^\^\^\^\^\^\^\^\^\^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaSyntaxError: Import not supported"
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
    #########################################################################


def test_disconnected_from_output__warning__format(capsys):
    def a0():
        print(1)
        return 1

    with warnings.catch_warnings(record=True) as w:
        parse(a0)

        wa = tuple(w[0].__dict__.values())[:6]

    myia_warning(*wa)

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): print\(1\)\n"
        + r"(.+?)  \^\^\^\^\^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaDisconnectedCodeWarning: "
        + r"Expression was not assigned to a variable\.\n"
        + r"\tAs a result, it is not connected to the output "
        + r"and will not be executed\."
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
    #########################################################################

    def a1():
        x = 1
        while x < 5:
            x = x + 1
        print(1)
        return 1

    with warnings.catch_warnings(record=True) as w:
        parse(a1)

        wa = tuple(w[0].__dict__.values())[:6]

    myia_warning(*wa)

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): print\(1\)\n"
        + r"(.+?)  \^\^\^\^\^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaDisconnectedCodeWarning: "
        + r"Expression was not assigned to a variable\.\n"
        + r"\tAs a result, it is not connected to the output "
        + r"and will not be executed\."
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
    #########################################################################

    def a2():
        def b2():
            return 1

        b2()
        return 1

    with warnings.catch_warnings(record=True) as w:
        parse(a2)

        wa = tuple(w[0].__dict__.values())[:6]

    myia_warning(*wa)

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): b2\(\)\n"
        + r"(.+?)  \^\^\^\^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaDisconnectedCodeWarning: "
        + r"Expression was not assigned to a variable\.\n"
        + r"\tAs a result, it is not connected to the output "
        + r"and will not be executed\."
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
    #########################################################################

    def a3():
        x = 1  # noqa: F841
        return 1

    with warnings.catch_warnings(record=True) as w:
        parse(a3)

        wa = tuple(w[0].__dict__.values())[:6]

    myia_warning(*wa)

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): x = 1  # noqa: F841\n"
        + r"(.+?)     \^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaDisconnectedCodeWarning: x is not used "
        + r"and will therefore not be computed"
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
    #########################################################################

    def a4():
        def b4():
            def c4():
                x = 1  # noqa: F841
                return 1

            return c4()

        return b4

    with warnings.catch_warnings(record=True) as w:
        parse(a4)

        wa = tuple(w[0].__dict__.values())[:6]

    myia_warning(*wa)

    out, err = capsys.readouterr()

    reg_pattern = (
        r"========================================"
        + r"========================================\n"
        + r"(.+?)/tests/test_parser\.py:(.+?)\n"
        + r"(.+?): x = 1  # noqa: F841\n"
        + r"(.+?)     \^\n"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        + r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        + r"MyiaDisconnectedCodeWarning: x is not used "
        + r"and will therefore not be computed"
    )

    regex = re.compile(reg_pattern)
    match = re.match(regex, err)

    assert match is not None
