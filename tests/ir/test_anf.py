
import pytest

from myia.ir.anf import PARAMETER, Apply, Constant, Graph, Parameter
from myia.prim import ops as primops


def test_incoming():
    in0 = Constant(0)
    value = Apply([in0], Graph())
    assert list(value.incoming) == [in0]
    assert list(in0.incoming) == []


def test_graph():
    """Construct a small graph.

    Note that this graph is strictly speaking nonsensical, because it doesn't
    use any actual primitive operations.
    """
    g = Graph()
    x = Parameter(g)
    assert x.value is PARAMETER
    one = Constant(1)
    add = Constant('add')
    return_ = Constant('return')
    value = Apply([add, x, one], g)
    return_ = Apply([return_, value], g)
    g.return_ = return_
    g.parameters.append(x)


def test_graph_helpers():
    """Test the helper methods on graphs."""
    g = Graph()
    x = g.add_parameter()
    y = g.add_parameter()
    assert g.parameters == [x, y]
    one = g.constant(1)
    add = g.constant('add')
    temp = g.apply('mul', one, 2)
    assert temp.graph is g
    assert all(isinstance(x, Constant) for x in temp.inputs)
    assert list(x.value for x in temp.inputs) == ['mul', 1, 2]
    g.output = g.apply(add, temp, x)
    assert g.output.graph is g
    assert list(g.output.inputs) == [add, temp, x]


def test_graph_output():
    g = Graph()
    with pytest.raises(Exception):
        print(g.output)
    one = Constant(1)
    g.output = one
    assert g.output is one
    assert isinstance(g.return_, Apply) and \
        len(g.return_.inputs) == 2 and \
        isinstance(g.return_.inputs[0], Constant) and \
        g.return_.inputs[0].value is primops.return_ and \
        g.return_.inputs[1] is one
    old_return = g.return_
    two = Constant(2)
    g.output = two
    assert g.return_ is old_return
    assert g.return_.inputs[1] is two


def test_str_coverage():
    """Just a coverage test for __str__ and __repr__

    Doesn't check that they take particular values since that could change
    easily.
    """
    g = Graph()
    p = Parameter(g)
    p.name = 'param'
    objects = [g, Apply([], g), p, Parameter(g), Constant(0), Constant(g)]
    for o in objects:
        str(o)
        repr(o)
        o.debug.debug_name
