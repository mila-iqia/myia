import math
import operator
from snektalk import pastevar

from myia.ir.visualization import GraphPrinter
from myia.parser import parse
from myia.utils.info import enable_debug


def f(x):
    while x:
        x = x - 1
    return x


def g():
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



def visualize(function):
    with enable_debug():
        graph = parse(function)
    print(GraphPrinter(graph, on_node=pastevar))


def main():
    visualize(g)
    # from hrepr import hrepr
    # hrepr.page(graph, file="output.html")


if __name__ == "__main__":
    main()
