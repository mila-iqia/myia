import pytest

from myia.api import parse
from myia.prim.signatures import MetaVar
from myia.analyze import GraphAnalyzer
from myia.dtype import Int


def test_MetaVar():
    mv = MetaVar()
    with pytest.raises(NotImplementedError):
        mv.infer((), None, None)


def test_getitem():
    def f(i):
        x = (1, 2.0, False)
        return x[0]

    gf = parse(f)
    ga = GraphAnalyzer()
    ga.analyze(gf)

    f_t = ga.signatures[gf]
    assert f_t.retval is Int(64)


def test_getitem2():
    def f(i):
        x = (1, 2.0, False)
        return x[1.0]

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)


def test_getitem3():
    def f(i):
        x = (1, 2.0, False)
        return x[-2]

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(ValueError):
        ga.analyze(gf)


def test_getitem4():
    def f(i):
        x = (1, 2.0, False)
        return x[i]

    gf = parse(f)
    ga = GraphAnalyzer()
    with pytest.raises(TypeError):
        ga.analyze(gf)
