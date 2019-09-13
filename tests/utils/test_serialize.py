import tempfile

import pytest

from myia import xtype
from myia.abstract import ANYTHING
from myia.ir import Constant, Graph, isomorphic
from myia.prim.ops import scalar_add, switch
from myia.utils import dump, load

from ..common import to_abstract_test

parametrize = pytest.mark.parametrize


def dumpload(o):
    f = tempfile.TemporaryFile()
    pos = f.tell()
    dump(o, f.fileno())
    f.seek(pos)
    return load(f.fileno())


@parametrize('v', [
    'potato',
    [],
    4242,
    (22,),
])
def test_roundtrip(v):
    v2 = dumpload(v)
    assert v is not v2
    assert v == v2


@parametrize('v', [
    None,
    (),
    22,
    switch,
    ANYTHING,
    xtype.Bool,
    xtype.i8,
    xtype.u64,
    xtype.f32,
])
def test_same(v):
    v2 = dumpload(v)
    assert v is v2


def test_dump_undefined():
    with pytest.raises(Exception):
        dumpload(object())


def test_exception():
    e2 = dumpload(Exception("this is bad"))
    assert e2.message == "Exception: this is bad\n"
    assert repr(e2) == 'LoadedException'


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
    node2 = dumpload(node)
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
    g2 = dumpload(g)

    assert g is not g2
    assert isinstance(g2, Graph)
    assert isomorphic(g, g2)
    assert g2.parameters[0].graph is g2
