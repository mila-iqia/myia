from myia.api import parse
from myia.analyze import GraphAnalyzer


def test_simple():
    def f():
        return 4 + 5

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    assert ga.values[gf.return_] == 9


def test_condt():
    def f():
        if True:
            return 3
        else:
            return 2

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    assert ga.values[gf.return_] == 3


def test_condf():
    def f():
        if False:
            return 3
        else:
            return 2

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    assert ga.values[gf.return_] == 2


def test_loop():

    def f():
        x = 2
        while x > 0:
            x = x - 1
        return x

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    assert ga.values[gf.return_] == 0


def test_mul():
    def f(x):
        return x * 0

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    assert ga.values[gf.return_] == 0
