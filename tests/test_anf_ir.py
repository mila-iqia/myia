import copy

import pytest

from myia.anf_ir import Graph, Apply, Parameter, Constant, PARAMETER, RETURN


def test_init_inputs():
    in0 = Constant(0)
    value = Apply([in0], Graph())
    assert in0.uses == {(value, 0)}


def test_append_inputs():
    in0 = Constant(0)
    value = Apply([], Graph())
    value.inputs.append(in0)
    assert in0.uses == {(value, 0)}


@pytest.mark.parametrize('index', [0, -1])
def test_insert_inputs(index):
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in1], Graph())
    value.inputs.insert(index, in0)
    assert value.inputs[0] is in0
    assert value.inputs[1] is in1
    assert in0.uses == {(value, 0)}
    assert in1.uses == {(value, 1)}


@pytest.mark.parametrize('index', [0, -1])
def test_get_inputs(index):
    in0 = Constant(0)
    value = Apply([in0], Graph())
    assert value.inputs[index] == in0


@pytest.mark.parametrize('index', [0, -1])
def test_set_inputs(index):
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0], Graph())
    value.inputs[index] = in1
    assert value.inputs[0] is in1
    assert in0.uses == set()
    assert in1.uses == {(value, 0)}


def test_incoming():
    in0 = Constant(0)
    value = Apply([in0], Graph())
    assert list(value.incoming) == [in0]
    assert list(in0.incoming) == []


def test_copy():
    in0 = Constant(0)
    value0 = Apply([in0], Graph())
    value1 = copy.copy(value0)
    in1 = copy.copy(in0)
    assert value1.inputs == value0.inputs
    assert value1.inputs is not value0.inputs
    assert in0.uses == {(value0, 0), (value1, 0)}
    assert in1.uses == set()


def test_outgoing():
    in0 = Constant(0)
    value = Apply([in0], Graph())
    assert list(value.outgoing) == []
    assert list(in0.outgoing) == [value]


@pytest.mark.parametrize('index', [0, -2])
def test_del_inputs(index):
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    del value.inputs[index]
    assert in0.uses == set()
    assert in1.uses == {(value, 0)}
    assert value.inputs[0] is in1


def test_slice_inputs():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    assert value.inputs[:] == [in0, in1]
    with pytest.raises(ValueError):
        del value.inputs[:]
    with pytest.raises(ValueError):
        value.inputs[:] = [in0]


def test_len_inputs():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    assert len(value.inputs) == 2


def test_repr_inputs():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    assert repr(value.inputs)
    assert str(value.inputs)


def test_set_inputs_property():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0], Graph())
    value.inputs = [in1]
    assert in0.uses == set()
    assert in1.uses == {(value, 0)}


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
        g.return_.inputs[0].value is RETURN and \
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
