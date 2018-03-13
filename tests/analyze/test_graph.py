import pytest

from myia.anf_ir import ANFNode, Apply, Constant, Graph
from myia import primops as P
from myia.api import parse

from myia.analyze.graph import GraphAnalyzer, Plugin, is_return, PluginManager


def test_is_return():
    v1 = Apply([Constant(P.return_), Constant(1)], Graph())
    v2 = Apply([Constant(Graph()), Constant(1)], Graph())
    v3 = Apply([Apply([], Graph()), Constant(1)], Graph())
    assert is_return(v1)
    assert not is_return(v2)
    assert not is_return(Constant(2))
    assert not is_return(v3)


def test_Plugin():
    # This is for coverage only since it is a dumb test
    p = Plugin()
    p.on_attach()


def test_PLuginManager():
    pp = PluginManager()
    assert len(pp) == 0

    class TPlugin(Plugin):
        NAME = "T"

    pp.add(TPlugin())
    assert len(pp) == 1

    T = TPlugin()

    with pytest.raises(AssertionError):
        pp.add(T)

    T.NAME = "on_node"
    with pytest.raises(AssertionError):
        pp.add(T)

    T.NAME = "_bad"
    with pytest.raises(AssertionError):
        pp.add(T)

    assert len(pp) == 1


def test_GraphAnaylzer():
    class DepPlugin(Plugin):
        NAME = "Dep"

        def on_attach(self):
            self.analyzer.some_attr = "set"

    class TPlugin(Plugin):
        NAME = "T"

        def __init__(self):
            self.count = 0
            self.active = False

        def on_attach(self):
            self.analyzer.add_plugin(DepPlugin())

            assert self.analyzer.some_attr == "set"

        def on_preprocess(self):
            assert not self.active
            self.active = True

        def on_node(self, node):
            assert isinstance(node, ANFNode)

            for i in node.inputs:
                assert i in self.analyzer._info_map

            assert node not in self.analyzer._info_map

            self.count += 1
            return self.count

        def on_graph(self, graph):
            assert isinstance(graph, Graph)

            for p in graph.parameters:
                assert p in self.analyzer._info_map

            assert graph not in self.analyzer.graphs

        def on_postprocess(self):
            assert self.active
            self.active = False

        def visit(self, fn, value):
            raise self.analyzer.DU.VisitError

    ga = GraphAnalyzer([TPlugin()])
    ga.add_plugin(DepPlugin())
    assert len(ga.plugins) == 2

    class TPlugin2(Plugin):
        NAME = "T"

    with pytest.raises(AssertionError):
        ga.add_plugin(TPlugin2())

    with pytest.raises(ValueError):
        ga.add_shortcut('_analyze_node', lambda n: None)

    with pytest.raises(ValueError):
        ga.add_shortcut('plugins', None)

    ga.add_shortcut('foo', 42)
    assert ga.foo == 42

    with pytest.raises(AttributeError):
        ga.bar

    with pytest.raises(ValueError):
        ga.add_shortcut('foo', None)

    def g(x, y):
        if x > y:
            return x
        else:
            return y

    def f(x):
        return g(x + 1, x + 2)

    gf = parse(f)

    ga.analyze(gf)

    assert len(ga.graphs) == 4
    assert len(ga._info_map) == 24

    assert all("T" in pp for pp in ga._info_map.values())
    assert all("T" in pp for pp in ga.graphs.values())

    cst = Constant(3)
    ga.analyze_node(cst)
    assert cst in ga._info_map

    ga.add_plugin(DepPlugin())
    assert len(ga.plugins) == 2

    class TPlugin3(Plugin):
        NAME = "T3"

    with pytest.raises(AssertionError):
        ga.add_plugin(TPlugin2())
