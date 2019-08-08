import io

import pytest

from myia.utils import load, dump
from myia.prim.ops import switch, scalar_add
from myia.abstract import ANYTHING
from myia import dtype
from myia.ir import Constant, Parameter, Apply, Graph, isomorphic

from ..common import to_abstract_test

parametrize = pytest.mark.parametrize


def dumpstr(o):
    stream = io.StringIO()
    dump(o, stream)
    return stream.getvalue()


def loadstr(s):
    stream = io.StringIO(s)
    return load(stream)


@parametrize('v', [
    'potato',
    [],
    4242,
    (22,),
])
def test_roundtrip(v):
    s = dumpstr(v)
    v2 = loadstr(s)
    assert v is not v2
    assert v == v2


@parametrize('v', [
    None,
    (),
    22,
    switch,
    ANYTHING,
    dtype.Bool,
    dtype.i8,
    dtype.u64,
    dtype.f32,
])
def test_same(v):
    s = dumpstr(v)
    v2 = loadstr(s)
    assert v is v2

g = Graph()
p1 = g.add_parameter()
p1.abstract = to_abstract_test(2)
p2 = g.add_parameter()
p2.abstract = to_abstract_test(2)
mid = g.apply(scalar_add, p1, p2)
mid.abstract = to_abstract_test(4)
c = Constant(1)
c.abstract = to_abstract_test(1)
g.output = g.apply(scalar_add, mid, c)
g.output.abstract = to_abstract_test(5)
g.return_.abstract = to_abstract_test(5)
g.return_.inputs[0].abstract = None


@parametrize('node', [
    p1,
    c,
    mid,
])
def test_anfnode(node):
    s = dumpstr(node)
    node2 = loadstr(s)
    assert type(node) is type(node2)
    assert len(node.inputs) == len(node2.inputs)
    # more inputs assert?
    if node.graph is None:
        assert node2.graph is None
    else:
        assert isinstance(node2.graph, Graph) and node.graph is not node2.graph
    assert node.debug.name == node2.debug.name
    assert node.debug.id == node2.debug.id
    # This is because of intern()
    assert node.abstract is node2.abstract


def test_graph():
    s = dumpstr(g)
    g2 = loadstr(s)

    assert g is not g2
    assert isinstance(g2, Graph)
    assert isomorphic(g, g2)
    assert g2.parameters[0].graph is g2
