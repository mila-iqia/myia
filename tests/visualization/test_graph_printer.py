import math
import operator
import os

import pytest
from hrepr import hrepr

from myia.ir.visualization import GraphPrinter
from myia.parser import parse
from myia.utils.info import enable_debug


def f0(x):
    while x:
        x = x - 1
    return x


def f1():
    a = 3
    b = -2.33
    c = -1.44e-9
    d = True
    e = "a string"
    g = (5, 7.7, -1, False)
    h = [5, 7.7, -1, False]
    i = {}
    j = {"a": "1", True: 2}
    k = dict()
    m = dict(a=1, b=2)
    n = operator.add
    p = math.sin
    return a, b, c, d, e, g, h, i, j, k, m, n, p


@pytest.mark.parametrize("link", (0, 1))
@pytest.mark.parametrize("show_args", (0, 1))
@pytest.mark.parametrize("show_fn_constants", (0, 1))
@pytest.mark.parametrize("function", (f0, f1))
def test_graph_printer(function, show_fn_constants, show_args, link):
    # Test by comparing generated hrepr output to expected html output.

    with enable_debug():
        graph = parse(function)
    gp = GraphPrinter(
        graph,
        show_args=bool(show_args),
        show_fn_constants=bool(show_fn_constants),
        link_fn_graphs=bool(link),
        link_inp_graphs=bool(link),
    )
    html = hrepr.page(gp)
    expected_filename = (
        f"{function.__name__}_{show_fn_constants}{show_args}{link}.html"
    )
    expected = open(
        os.path.join(os.path.dirname(__file__), expected_filename)
    ).read()
    assert str(html) == expected
