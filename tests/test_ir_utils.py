from myia.anf_ir import Constant, Apply, Graph
from myia.ir_utils import dfs


def test_dfs():
    in0 = Constant(0)
    in1 = Constant(1)
    value = Apply([in0, in1], Graph())
    assert next(dfs(value)) == value
    assert set(dfs(value)) == {value, in0, in1}
