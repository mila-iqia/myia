import pytest

from myia.ir import (Graph, Apply, Return, Parameter, Constant,
                     PARAMETER, RETURN)


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
    assert str(value.inputs) == f"Inputs({repr([in0, in1])})"


def test_set_inputs_property():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0], Graph())
    value.inputs = [in1]
    assert in0.uses == set()
    assert in1.uses == {(value, 0)}


def test_graph():
    g = Graph()
    x = Parameter(g)
    assert x.value is PARAMETER
    one = Constant(1)
    add = Constant('add')
    value = Apply([add, x, one], g)
    return_ = Return(value, g)
    assert return_.value is RETURN
    g.return_ = return_
    g.parameters.append(x)
